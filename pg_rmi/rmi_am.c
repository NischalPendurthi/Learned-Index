// Access method handler, reloptions, model cache, planner costs, rmi_stats().
#include "rmi.h"

#include "access/htup_details.h"
#include "access/reloptions.h"
#include "access/relation.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/selfuncs.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(rmi_handler);
PG_FUNCTION_INFO_V1(rmi_stats);

static relopt_kind rmi_relopt_kind;

void		_PG_init(void);

void
_PG_init(void)
{
	rmi_relopt_kind = add_reloption_kind();
	add_int_reloption(rmi_relopt_kind, "fillfactor",
					  "Percent of each leaf page filled at build, the rest is room for inserts",
					  RMI_DEFAULT_FILLFACTOR, 10, 100, AccessExclusiveLock);
	add_int_reloption(rmi_relopt_kind, "models",
					  "Number of stage-2 models, 0 chooses from the number of rows",
					  0, 0, 10000000, AccessExclusiveLock);
}

bytea *
rmi_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"fillfactor", RELOPT_TYPE_INT, offsetof(RmiOptions, fillfactor)},
		{"models", RELOPT_TYPE_INT, offsetof(RmiOptions, models)},
	};

	return (bytea *) build_reloptions(reloptions, validate, rmi_relopt_kind,
									  sizeof(RmiOptions), tab, lengthof(tab));
}

// Reads the meta page and every model page once per backend, so a lookup can
// predict its leaf without touching a buffer.
RmiCache *
rmi_get_cache(Relation index)
{
	RmiCache   *cache = (RmiCache *) index->rd_amcache;
	Buffer		buf;
	Page		page;
	RmiMetaData meta;
	Size		knot_bytes,
				model_bytes,
				total,
				copied = 0;
	char	   *dest;
	BlockNumber blk;

	if (cache)
		return cache;

	buf = ReadBuffer(index, RMI_META_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	memcpy(&meta, PageGetContents(page), sizeof(RmiMetaData));
	UnlockReleaseBuffer(buf);

	if (meta.magic != RMI_MAGIC || meta.version != RMI_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("index \"%s\" is not a valid rmi index",
						RelationGetRelationName(index))));

	knot_bytes = sizeof(RmiKnot) * meta.nknots;
	model_bytes = sizeof(RmiLinear) * meta.nmodels;
	total = knot_bytes + model_bytes;

	cache = MemoryContextAlloc(index->rd_indexcxt, MAXALIGN(sizeof(RmiCache)) + total + 1);
	cache->meta = meta;
	dest = (char *) cache + MAXALIGN(sizeof(RmiCache));
	cache->knots = (RmiKnot *) dest;
	cache->models = (RmiLinear *) (dest + knot_bytes);

	blk = meta.first_model_blk;
	while (copied < total)
	{
		Size		n = Min(total - copied, RMI_CONTENT_SIZE);

		buf = ReadBuffer(index, blk++);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		memcpy(dest + copied, PageGetContents(BufferGetPage(buf)), n);
		UnlockReleaseBuffer(buf);
		copied += n;
	}

	index->rd_amcache = cache;
	return cache;
}

static void
rmi_costestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				 Cost *startup, Cost *total, Selectivity *sel,
				 double *correlation, double *pages)
{
	GenericCosts costs = {0};

	genericcostestimate(root, path, loop_count, &costs);

	*startup = costs.indexStartupCost;
	*total = costs.indexTotalCost;
	*sel = costs.indexSelectivity;
	*correlation = costs.indexCorrelation;
	*pages = costs.numIndexPages;
}

static bool
rmi_validate(Oid opclassoid)
{
	return true;
}

Datum
rmi_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *am = makeNode(IndexAmRoutine);

	am->amstrategies = 5;
	am->amsupport = 1;
	am->amoptsprocnum = 0;
	// Overflow entries are not in key order relative to the leaf, so the index
	// cannot return rows sorted.
	am->amcanorder = false;
	am->amcanorderbyop = false;
	am->amcanhash = false;
	am->amconsistentequality = false;
	am->amconsistentordering = false;
	am->amcanbackward = false;
	am->amcanunique = false;
	am->amcanmulticol = false;
	// A qual on the column is required and NULLs are not stored.
	am->amoptionalkey = false;
	am->amsearcharray = false;
	am->amsearchnulls = false;
	am->amstorage = false;
	am->amclusterable = false;
	am->ampredlocks = false;
	am->amcanparallel = false;
	am->amcanbuildparallel = false;
	am->amcaninclude = false;
	am->amusemaintenanceworkmem = false;
	am->amsummarizing = false;
	am->amparallelvacuumoptions = 0;
	am->amkeytype = INT8OID;

	am->ambuild = rmi_build;
	am->ambuildempty = rmi_buildempty;
	am->aminsert = rmi_insert;
	am->aminsertcleanup = NULL;
	am->ambulkdelete = rmi_bulkdelete;
	am->amvacuumcleanup = rmi_vacuumcleanup;
	am->amcanreturn = NULL;
	am->amcostestimate = rmi_costestimate;
	am->amgettreeheight = NULL;
	am->amoptions = rmi_options;
	am->amproperty = NULL;
	am->ambuildphasename = NULL;
	am->amvalidate = rmi_validate;
	am->amadjustmembers = NULL;
	am->ambeginscan = rmi_beginscan;
	am->amrescan = rmi_rescan;
	am->amgettuple = rmi_gettuple;
	am->amgetbitmap = rmi_getbitmap;
	am->amendscan = rmi_endscan;
	am->ammarkpos = NULL;
	am->amrestrpos = NULL;
	am->amestimateparallelscan = NULL;
	am->aminitparallelscan = NULL;
	am->amparallelrescan = NULL;
	am->amtranslatestrategy = NULL;
	am->amtranslatecmptype = NULL;

	PG_RETURN_POINTER(am);
}

// rmi_stats(index): model quality and how much the index has drifted.
Datum
rmi_stats(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	Relation	index = relation_open(relid, AccessShareLock);
	TupleDesc	tupdesc;
	RmiCache   *cache;
	BlockNumber nblocks;
	int64		leaf_entries = 0,
				ovf_entries = 0,
				ovf_pages = 0,
				leaves_with_chain = 0;
	Datum		values[12];
	bool		nulls[12] = {0};
	static const char *const kinds[] = {"linear", "loglinear", "piecewise"};

	if (index->rd_rel->relkind != RELKIND_INDEX || index->rd_indam->ambuild != rmi_build)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an rmi index", RelationGetRelationName(index))));

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	cache = rmi_get_cache(index);
	nblocks = RelationGetNumberOfBlocks(index);

	for (BlockNumber blk = cache->meta.first_leaf_blk; blk < nblocks; blk++)
	{
		Buffer		buf = ReadBuffer(index, blk);
		RmiPageOpaque op;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		op = RmiOpaque(BufferGetPage(buf));
		if (op->flags & RMI_F_LEAF)
		{
			leaf_entries += op->nentries;
			if (BlockNumberIsValid(op->next))
				leaves_with_chain++;
		}
		else if (op->flags & RMI_F_OVERFLOW)
		{
			ovf_entries += op->nentries;
			ovf_pages++;
		}
		UnlockReleaseBuffer(buf);
	}

	values[0] = Int64GetDatum(cache->meta.nentries);
	values[1] = Int32GetDatum(cache->meta.nmodels);
	values[2] = CStringGetTextDatum(kinds[cache->meta.stage1_kind]);
	values[3] = Int32GetDatum(cache->meta.nknots);
	values[4] = Float8GetDatum(cache->meta.avg_abs_err);
	values[5] = Float8GetDatum(cache->meta.avg_window);
	values[6] = Int64GetDatum(cache->meta.max_window);
	values[7] = Int64GetDatum((int64) cache->meta.nleaves);
	values[8] = Int64GetDatum(leaf_entries);
	values[9] = Int64GetDatum(ovf_pages);
	values[10] = Int64GetDatum(ovf_entries);
	values[11] = Float8GetDatum(cache->meta.nleaves ? (double) ovf_pages / cache->meta.nleaves : 0.0);

	relation_close(index, AccessShareLock);

	(void) leaves_with_chain;
	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}
