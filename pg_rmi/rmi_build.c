// CREATE INDEX / REINDEX: read the table, sort, train, write the pages.
#include "rmi.h"

#include "access/tableam.h"
#include "storage/bulk_write.h"
#include "utils/memutils.h"

typedef struct BuildState
{
	RmiEntry   *entries;
	int64		n;
	int64		cap;
} BuildState;

static void
build_callback(Relation index, ItemPointer tid, Datum *values, bool *isnull,
			   bool tupleIsAlive, void *state)
{
	BuildState *bs = (BuildState *) state;

	if (isnull[0])
		return;
	if (bs->n == bs->cap)
	{
		bs->cap *= 2;
		bs->entries = repalloc_huge(bs->entries, sizeof(RmiEntry) * bs->cap);
	}
	bs->entries[bs->n].key = DatumGetInt64(values[0]);
	bs->entries[bs->n].tid = *tid;
	bs->n++;
}

static int
entry_cmp(const void *a, const void *b)
{
	const RmiEntry *x = a;
	const RmiEntry *y = b;

	if (x->key != y->key)
		return x->key < y->key ? -1 : 1;
	return ItemPointerCompare((ItemPointer) &x->tid, (ItemPointer) &y->tid);
}

static void
init_page(Page page, uint16 flags)
{
	RmiPageOpaque op;

	PageInit(page, BLCKSZ, sizeof(RmiPageOpaqueData));
	op = RmiOpaque(page);
	op->low = PG_INT64_MIN;
	op->high = PG_INT64_MAX;
	op->next = InvalidBlockNumber;
	op->nentries = 0;
	op->flags = flags;
	op->page_id = RMI_PAGE_ID;
}

static void
write_meta(BulkWriteState *bulk, const RmiMetaData *meta)
{
	BulkWriteBuffer buf = smgr_bulk_get_buf(bulk);
	Page		page = (Page) buf;

	init_page(page, RMI_F_META);
	memcpy(PageGetContents(page), meta, sizeof(RmiMetaData));
	((PageHeader) page)->pd_lower =
		(char *) PageGetContents(page) + sizeof(RmiMetaData) - (char *) page;
	smgr_bulk_write(bulk, RMI_META_BLKNO, buf, true);
}

// A leaf or overflow page's whole array area counts as in use: entries are kept
// at fixed offsets, so there is no hole for WAL to skip.
void		rmi_set_page_bounds(Page page);

void
rmi_set_page_bounds(Page page)
{
	((PageHeader) page)->pd_lower =
		(char *) RmiTids(page) + RMI_LEAF_CAPACITY * sizeof(ItemPointerData) - (char *) page;
}

// Meta page plus one empty leaf covering every key, so inserts have a target.
static void
write_empty(Relation index, ForkNumber fork)
{
	BulkWriteState *bulk = smgr_bulk_start_rel(index, fork);
	RmiMetaData meta = {0};
	BulkWriteBuffer buf;
	Page		page;

	meta.magic = RMI_MAGIC;
	meta.version = RMI_VERSION;
	meta.per_leaf = RMI_LEAF_CAPACITY;
	meta.first_model_blk = 1;
	meta.first_leaf_blk = 1;
	meta.nleaves = 1;
	write_meta(bulk, &meta);

	buf = smgr_bulk_get_buf(bulk);
	page = (Page) buf;
	init_page(page, RMI_F_LEAF);
	rmi_set_page_bounds(page);
	smgr_bulk_write(bulk, 1, buf, true);

	smgr_bulk_finish(bulk);
}

void
rmi_buildempty(Relation index)
{
	write_empty(index, INIT_FORKNUM);
}

IndexBuildResult *
rmi_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result = palloc(sizeof(IndexBuildResult));
	RmiOptions *opts = (RmiOptions *) index->rd_options;
	int			fillfactor = opts ? opts->fillfactor : RMI_DEFAULT_FILLFACTOR;
	int			models = opts ? opts->models : 0;
	BuildState	bs;
	double		heap_tuples;
	int			per_leaf;
	uint32		nmodels;
	RmiTrained	trained;
	RmiMetaData *meta = &trained.meta;
	BulkWriteState *bulk;
	Size		knot_bytes,
				model_bytes,
				total,
				written = 0;
	char	   *payload;
	BlockNumber blk;
	BlockNumber model_pages;
	int64		nleaves;

	if (RelationGetNumberOfBlocks(index) != 0)
		elog(ERROR, "index \"%s\" already contains data", RelationGetRelationName(index));

	bs.cap = Max(1024, (int64) heap->rd_rel->reltuples + 1);
	bs.n = 0;
	bs.entries = MemoryContextAllocHuge(CurrentMemoryContext, sizeof(RmiEntry) * bs.cap);

	heap_tuples = table_index_build_scan(heap, index, indexInfo, true, true,
										 build_callback, &bs, NULL);

	if (bs.n == 0)
	{
		write_empty(index, MAIN_FORKNUM);
		pfree(bs.entries);
		result->heap_tuples = heap_tuples;
		result->index_tuples = 0;
		return result;
	}

	qsort(bs.entries, bs.n, sizeof(RmiEntry), entry_cmp);

	per_leaf = Max(1, RMI_LEAF_CAPACITY * fillfactor / 100);
	nmodels = models > 0 ? (uint32) models : (uint32) Max(1, Min(bs.n / 1000, 100000));

	memset(&trained, 0, sizeof(trained));
	rmi_train(bs.entries, bs.n, nmodels, &trained);

	knot_bytes = sizeof(RmiKnot) * meta->nknots;
	model_bytes = sizeof(RmiLinear) * meta->nmodels;
	total = knot_bytes + model_bytes;
	model_pages = (BlockNumber) ((total + RMI_CONTENT_SIZE - 1) / RMI_CONTENT_SIZE);
	nleaves = (bs.n + per_leaf - 1) / per_leaf;

	meta->magic = RMI_MAGIC;
	meta->version = RMI_VERSION;
	meta->per_leaf = per_leaf;
	meta->first_model_blk = 1;
	meta->first_leaf_blk = 1 + model_pages;
	meta->nleaves = (BlockNumber) nleaves;

	bulk = smgr_bulk_start_rel(index, MAIN_FORKNUM);
	write_meta(bulk, meta);

	payload = palloc(total);
	memcpy(payload, trained.knots, knot_bytes);
	memcpy(payload + knot_bytes, trained.models, model_bytes);

	blk = meta->first_model_blk;
	while (written < total)
	{
		BulkWriteBuffer buf = smgr_bulk_get_buf(bulk);
		Page		page = (Page) buf;
		Size		n = Min(total - written, RMI_CONTENT_SIZE);

		init_page(page, RMI_F_MODEL);
		memcpy(PageGetContents(page), payload + written, n);
		((PageHeader) page)->pd_lower = (char *) PageGetContents(page) + n - (char *) page;
		smgr_bulk_write(bulk, blk++, buf, true);
		written += n;
	}
	pfree(payload);

	for (int64 l = 0; l < nleaves; l++)
	{
		BulkWriteBuffer buf = smgr_bulk_get_buf(bulk);
		Page		page = (Page) buf;
		RmiPageOpaque op;
		int64		first = l * per_leaf;
		int			count = (int) Min(per_leaf, bs.n - first);
		int64	   *keys;
		ItemPointerData *tids;

		init_page(page, RMI_F_LEAF);
		rmi_set_page_bounds(page);
		op = RmiOpaque(page);
		keys = RmiKeys(page);
		tids = RmiTids(page);

		for (int i = 0; i < count; i++)
		{
			keys[i] = bs.entries[first + i].key;
			tids[i] = bs.entries[first + i].tid;
		}
		op->nentries = (uint16) count;

		// The first leaf takes everything below it and the last everything
		// above it, including keys inserted later.
		if (l > 0)
		{
			op->low = keys[0];
			if (bs.entries[first - 1].key == keys[0])
				op->flags |= RMI_F_DUP_LEFT;
		}
		if (l < nleaves - 1)
		{
			op->high = bs.entries[first + count].key;
			if (bs.entries[first + count].key == keys[count - 1])
				op->flags |= RMI_F_DUP_RIGHT;
		}

		smgr_bulk_write(bulk, blk++, buf, true);
	}

	smgr_bulk_finish(bulk);

	result->heap_tuples = heap_tuples;
	result->index_tuples = (double) bs.n;

	pfree(bs.entries);
	pfree(trained.models);
	pfree(trained.knots);
	return result;
}
