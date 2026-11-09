// Finding a leaf with the model, and equality / range scans.
#include "rmi.h"

#include "access/relscan.h"
#include "access/stratnum.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "nodes/tidbitmap.h"
#include "pgstat.h"

typedef struct RmiScanOpaqueData
{
	bool		started;
	bool		done;
	bool		empty;			// the quals can't match anything
	int64		lo;				// inclusive
	int64		hi;				// inclusive

	BlockNumber leaf_no;		// current leaf, 0-based among the leaves
	int64		leaf_high;		// its high fence
	BlockNumber next_overflow;	// next page of its chain, if any
	Buffer		pinned;			// held while its TIDs are being returned

	ItemPointerData items[MaxHeapTuplesPerPage > RMI_LEAF_CAPACITY ? MaxHeapTuplesPerPage : RMI_LEAF_CAPACITY];
	int			nitems;
	int			cur;
} RmiScanOpaqueData;

typedef RmiScanOpaqueData *RmiScanOpaque;

// -1: key belongs to an earlier leaf, 1: to a later one, 0: this one. Where a
// run of equal keys crosses a boundary, the leftmost leaf of the run owns it.
static inline int
leaf_direction(RmiPageOpaque op, int64 key, bool is_first, bool is_last)
{
	if (!is_first && (key < op->low || (key == op->low && (op->flags & RMI_F_DUP_LEFT))))
		return -1;
	if (!is_last && (key > op->high || (key == op->high && !(op->flags & RMI_F_DUP_RIGHT))))
		return 1;
	return 0;
}

// Returns the leaf owning key, locked in lockmode. The model and its error
// bounds choose which pages to read. The fences on each page decide whether we
// are in the right place, so a bad prediction costs extra reads, never a wrong
// answer.
Buffer
rmi_locate_leaf(Relation index, RmiCache *cache, int64 key, int lockmode, BlockNumber *leaf_no)
{
	const RmiMetaData *meta = &cache->meta;
	int64		last = (int64) meta->nleaves - 1;
	int64		lo = 0;
	int64		hi = last;
	int64		p = 0;

	if (meta->nentries > 0)
	{
		double		x = rmi_key_x(meta, key);
		const RmiLinear *m = &cache->models[rmi_model_index(meta, cache->knots, x)];
		int64		pos = rmi_predict_pos(meta, m, x);

		p = Min(pos / meta->per_leaf, last);
		lo = Max(0, (pos + m->min_err) / meta->per_leaf);
		hi = Min(last, (pos + m->max_err) / meta->per_leaf);
		lo = Min(lo, p);
		hi = Max(hi, p);
	}

	for (;;)
	{
		Buffer		buf = ReadBuffer(index, meta->first_leaf_blk + (BlockNumber) p);
		int			dir;

		LockBuffer(buf, lockmode);
		dir = leaf_direction(RmiOpaque(BufferGetPage(buf)), key, p == 0, p == last);
		if (dir == 0)
		{
			*leaf_no = (BlockNumber) p;
			return buf;
		}
		UnlockReleaseBuffer(buf);

		// The error bounds hold for keys present at build. An absent key can
		// sit just outside them, so if the window runs out, widen it.
		if (dir < 0)
		{
			hi = p - 1;
			if (lo > hi)
				lo = 0;
		}
		else
		{
			lo = p + 1;
			if (lo > hi)
				hi = last;
		}
		p = lo + (hi - lo) / 2;
		CHECK_FOR_INTERRUPTS();
	}
}

IndexScanDesc
rmi_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);
	RmiScanOpaque so = palloc0(sizeof(RmiScanOpaqueData));

	so->pinned = InvalidBuffer;
	scan->opaque = so;
	return scan;
}

static void
release_pin(RmiScanOpaque so)
{
	if (BufferIsValid(so->pinned))
	{
		ReleaseBuffer(so->pinned);
		so->pinned = InvalidBuffer;
	}
}

static int64
key_datum(ScanKey k)
{
	Oid			subtype = k->sk_subtype;

	if (subtype == INT4OID)
		return (int64) DatumGetInt32(k->sk_argument);
	if (subtype == INT2OID)
		return (int64) DatumGetInt16(k->sk_argument);
	return DatumGetInt64(k->sk_argument);
}

void
rmi_rescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	RmiScanOpaque so = (RmiScanOpaque) scan->opaque;

	release_pin(so);
	so->started = false;
	so->done = false;
	so->empty = false;
	so->lo = PG_INT64_MIN;
	so->hi = PG_INT64_MAX;
	so->nitems = 0;
	so->cur = 0;
	so->next_overflow = InvalidBlockNumber;

	if (keys && scan->numberOfKeys > 0)
		memcpy(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	for (int i = 0; i < scan->numberOfKeys; i++)
	{
		ScanKey		k = &scan->keyData[i];
		int64		v;

		if (k->sk_flags & SK_ISNULL)
		{
			so->empty = true;
			continue;
		}
		v = key_datum(k);

		switch (k->sk_strategy)
		{
			case BTLessStrategyNumber:
				if (v == PG_INT64_MIN)
					so->empty = true;
				else
					so->hi = Min(so->hi, v - 1);
				break;
			case BTLessEqualStrategyNumber:
				so->hi = Min(so->hi, v);
				break;
			case BTEqualStrategyNumber:
				so->lo = Max(so->lo, v);
				so->hi = Min(so->hi, v);
				break;
			case BTGreaterEqualStrategyNumber:
				so->lo = Max(so->lo, v);
				break;
			case BTGreaterStrategyNumber:
				if (v == PG_INT64_MAX)
					so->empty = true;
				else
					so->lo = Max(so->lo, v + 1);
				break;
			default:
				elog(ERROR, "unexpected strategy number %d", k->sk_strategy);
		}
	}
	if (so->lo > so->hi)
		so->empty = true;
}

// Copy this page's matching TIDs. Keeps the pin but drops the lock.
static void
read_page(RmiScanOpaque so, Buffer buf)
{
	Page		page = BufferGetPage(buf);
	RmiPageOpaque op = RmiOpaque(page);
	int64	   *keys = RmiKeys(page);
	ItemPointerData *tids = RmiTids(page);
	int			n = op->nentries;

	so->nitems = 0;
	so->cur = 0;
	for (int i = rmi_page_lower_bound(keys, n, so->lo); i < n && keys[i] <= so->hi; i++)
		so->items[so->nitems++] = tids[i];

	so->next_overflow = op->next;
	if (op->flags & RMI_F_LEAF)
		so->leaf_high = op->high;

	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	so->pinned = buf;
}

// Moves to the next page that has matches. Returns false when the scan is over.
static bool
next_batch(IndexScanDesc scan)
{
	RmiScanOpaque so = (RmiScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	RmiCache   *cache = rmi_get_cache(index);

	if (so->empty || so->done)
		return false;

	for (;;)
	{
		Buffer		buf;

		release_pin(so);
		CHECK_FOR_INTERRUPTS();

		if (!so->started)
		{
			so->started = true;
			if (so->lo == PG_INT64_MIN)
			{
				so->leaf_no = 0;
				buf = ReadBuffer(index, cache->meta.first_leaf_blk);
				LockBuffer(buf, BUFFER_LOCK_SHARE);
			}
			else
				buf = rmi_locate_leaf(index, cache, so->lo, BUFFER_LOCK_SHARE, &so->leaf_no);
			pgstat_count_index_scan(index);
		}
		else if (BlockNumberIsValid(so->next_overflow))
		{
			buf = ReadBuffer(index, so->next_overflow);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
		}
		else
		{
			// done with this leaf and its chain: does the range continue?
			if (so->leaf_no + 1 >= cache->meta.nleaves || so->leaf_high > so->hi)
			{
				so->done = true;
				return false;
			}
			so->leaf_no++;
			buf = ReadBuffer(index, cache->meta.first_leaf_blk + so->leaf_no);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
		}

		read_page(so, buf);
		if (so->nitems > 0)
			return true;
	}
}

bool
rmi_gettuple(IndexScanDesc scan, ScanDirection dir)
{
	RmiScanOpaque so = (RmiScanOpaque) scan->opaque;

	if (dir != ForwardScanDirection)
		elog(ERROR, "rmi index does not support backward scans");

	if (so->cur >= so->nitems && !next_batch(scan))
		return false;

	scan->xs_heaptid = so->items[so->cur++];
	scan->xs_recheck = false;
	return true;
}

int64
rmi_getbitmap(IndexScanDesc scan, TIDBitmap *tbm)
{
	RmiScanOpaque so = (RmiScanOpaque) scan->opaque;
	int64		total = 0;

	while (next_batch(scan))
	{
		tbm_add_tuples(tbm, so->items, so->nitems, false);
		total += so->nitems;
		so->cur = so->nitems;
	}
	release_pin(so);
	return total;
}

void
rmi_endscan(IndexScanDesc scan)
{
	RmiScanOpaque so = (RmiScanOpaque) scan->opaque;

	release_pin(so);
	pfree(so);
}
