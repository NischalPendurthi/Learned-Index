// Inserts and VACUUM. Every change goes through GenericXLog, so it is crash safe.
//
// A leaf never splits, because the model's position -> page mapping depends on
// that. When a leaf fills up, new entries go to its overflow chain. Lookups have
// to read the chain as well, so performance drifts until REINDEX retrains.
#include "rmi.h"

#include "access/htup_details.h"
#include "commands/vacuum.h"
#include "miscadmin.h"
#include "storage/indexfsm.h"

extern void rmi_set_page_bounds(Page page);

static void
page_add(Page page, int64 key, ItemPointer tid)
{
	RmiPageOpaque op = RmiOpaque(page);
	int64	   *keys = RmiKeys(page);
	ItemPointerData *tids = RmiTids(page);
	int			n = op->nentries;
	int			at = rmi_page_lower_bound(keys, n, key);

	memmove(keys + at + 1, keys + at, (n - at) * sizeof(int64));
	memmove(tids + at + 1, tids + at, (n - at) * sizeof(ItemPointerData));
	keys[at] = key;
	tids[at] = *tid;
	op->nentries = n + 1;
}

static void
xlog_add(Relation index, Buffer buf, int64 key, ItemPointer tid)
{
	GenericXLogState *state = GenericXLogStart(index);
	Page		page = GenericXLogRegisterBuffer(state, buf, 0);

	page_add(page, key, tid);
	GenericXLogFinish(state);
}

bool
rmi_insert(Relation index, Datum *values, bool *isnull, ItemPointer ht_ctid,
		   Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
		   IndexInfo *indexInfo)
{
	RmiCache   *cache;
	int64		key;
	BlockNumber leaf_no;
	Buffer		leaf;
	RmiPageOpaque op;
	BlockNumber head;
	Buffer		ovf;
	GenericXLogState *state;
	Page		leaf_page;
	Page		new_page;
	RmiPageOpaque new_op;

	if (isnull[0])
		return false;

	key = DatumGetInt64(values[0]);
	cache = rmi_get_cache(index);
	leaf = rmi_locate_leaf(index, cache, key, BUFFER_LOCK_EXCLUSIVE, &leaf_no);
	op = RmiOpaque(BufferGetPage(leaf));

	if (op->nentries < RMI_LEAF_CAPACITY)
	{
		xlog_add(index, leaf, key, ht_ctid);
		UnlockReleaseBuffer(leaf);
		return false;
	}

	// Leaf is full. Only the head of the chain can have room, since new pages
	// always go at the head. The leaf stays locked, which serializes inserts
	// into this chain.
	head = op->next;
	if (BlockNumberIsValid(head))
	{
		ovf = ReadBuffer(index, head);
		LockBuffer(ovf, BUFFER_LOCK_EXCLUSIVE);
		if (RmiOpaque(BufferGetPage(ovf))->nentries < RMI_LEAF_CAPACITY)
		{
			xlog_add(index, ovf, key, ht_ctid);
			UnlockReleaseBuffer(ovf);
			UnlockReleaseBuffer(leaf);
			return false;
		}
		UnlockReleaseBuffer(ovf);
	}

	// New chain head. Both pages change in one WAL record.
	ovf = ExtendBufferedRel(BMR_REL(index), MAIN_FORKNUM, NULL, EB_LOCK_FIRST);

	state = GenericXLogStart(index);
	new_page = GenericXLogRegisterBuffer(state, ovf, GENERIC_XLOG_FULL_IMAGE);
	leaf_page = GenericXLogRegisterBuffer(state, leaf, 0);

	PageInit(new_page, BLCKSZ, sizeof(RmiPageOpaqueData));
	new_op = RmiOpaque(new_page);
	new_op->low = PG_INT64_MIN;
	new_op->high = PG_INT64_MAX;
	new_op->next = head;
	new_op->nentries = 0;
	new_op->flags = RMI_F_OVERFLOW;
	new_op->page_id = RMI_PAGE_ID;
	rmi_set_page_bounds(new_page);
	page_add(new_page, key, ht_ctid);

	RmiOpaque(leaf_page)->next = BufferGetBlockNumber(ovf);

	GenericXLogFinish(state);

	UnlockReleaseBuffer(ovf);
	UnlockReleaseBuffer(leaf);
	return false;
}

// Removes dead TIDs. A scan keeps its page pinned while it returns that page's
// TIDs, and the cleanup lock waits for those pins, so a scan can never return a
// TID whose heap slot has been reused.
IndexBulkDeleteResult *
rmi_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
			   IndexBulkDeleteCallback callback, void *cb_state)
{
	Relation	index = info->index;
	RmiCache   *cache = rmi_get_cache(index);
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);

	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	for (BlockNumber blk = cache->meta.first_leaf_blk; blk < nblocks; blk++)
	{
		Buffer		buf;
		Page		page;
		RmiPageOpaque op;
		int			n,
					dead = 0;
		bool		is_dead[RMI_LEAF_CAPACITY];

		vacuum_delay_point(false);

		buf = ReadBufferExtended(index, MAIN_FORKNUM, blk, RBM_NORMAL, info->strategy);
		LockBufferForCleanup(buf);
		page = BufferGetPage(buf);
		op = RmiOpaque(page);

		if (PageIsNew(page) || !(op->flags & (RMI_F_LEAF | RMI_F_OVERFLOW)))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}

		n = op->nentries;
		for (int i = 0; i < n; i++)
		{
			is_dead[i] = callback(&RmiTids(page)[i], cb_state);
			if (is_dead[i])
				dead++;
		}

		if (dead > 0)
		{
			GenericXLogState *state = GenericXLogStart(index);
			Page		wpage = GenericXLogRegisterBuffer(state, buf, 0);
			int64	   *keys = RmiKeys(wpage);
			ItemPointerData *tids = RmiTids(wpage);
			int			kept = 0;

			for (int i = 0; i < n; i++)
			{
				if (is_dead[i])
					continue;
				keys[kept] = keys[i];
				tids[kept] = tids[i];
				kept++;
			}
			RmiOpaque(wpage)->nentries = kept;
			GenericXLogFinish(state);
		}

		stats->tuples_removed += dead;
		stats->num_index_tuples += n - dead;
		UnlockReleaseBuffer(buf);
	}

	stats->num_pages = nblocks;
	return stats;
}

IndexBulkDeleteResult *
rmi_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	Relation	index = info->index;
	BlockNumber nblocks;

	if (info->analyze_only)
		return stats;

	if (stats != NULL)
		return stats;

	// no dead tuples this round: still report accurate counts
	stats = palloc0(sizeof(IndexBulkDeleteResult));
	nblocks = RelationGetNumberOfBlocks(index);

	for (BlockNumber blk = RMI_META_BLKNO + 1; blk < nblocks; blk++)
	{
		Buffer		buf;
		RmiPageOpaque op;

		vacuum_delay_point(false);
		buf = ReadBufferExtended(index, MAIN_FORKNUM, blk, RBM_NORMAL, info->strategy);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		op = RmiOpaque(BufferGetPage(buf));
		if (!PageIsNew(BufferGetPage(buf)) && (op->flags & (RMI_F_LEAF | RMI_F_OVERFLOW)))
			stats->num_index_tuples += op->nentries;
		UnlockReleaseBuffer(buf);
	}
	stats->num_pages = nblocks;
	return stats;
}
