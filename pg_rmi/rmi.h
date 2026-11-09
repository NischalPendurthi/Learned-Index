// Learned index (2-stage RMI) as a PostgreSQL index access method.
//
// On disk:
//   block 0            meta page
//   blocks 1..m        model pages (piecewise knots, then stage-2 models)
//   next nleaves       leaf pages, sorted (key, TID), fixed number per page at build
//   anything after     overflow pages, created by inserts
//
// The model predicts a position in the sorted array as it was at build time,
// and position / per_leaf gives the leaf page. Leaves never split, so that
// mapping stays valid until REINDEX retrains the index.
#ifndef RMI_H
#define RMI_H

#include "postgres.h"

#include "access/amapi.h"
#include "access/genam.h"
#include "access/generic_xlog.h"
#include "nodes/execnodes.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/itemptr.h"
#include "utils/rel.h"

#define RMI_MAGIC 0x524D4931
#define RMI_VERSION 1
#define RMI_META_BLKNO 0
#define RMI_PAGE_ID 0xFF91

// page types
#define RMI_F_META 0x01
#define RMI_F_MODEL 0x02
#define RMI_F_LEAF 0x04
#define RMI_F_OVERFLOW 0x08
// on a leaf: equal keys continue across the page boundary
#define RMI_F_DUP_LEFT 0x10
#define RMI_F_DUP_RIGHT 0x20

typedef struct RmiPageOpaqueData
{
	int64		low;			// first key this leaf owns (fixed at build)
	int64		high;			// first key of the next leaf (fixed at build)
	BlockNumber next;			// head of the overflow chain, or next in it
	uint16		nentries;
	uint16		flags;
	uint16		page_id;
} RmiPageOpaqueData;

typedef RmiPageOpaqueData *RmiPageOpaque;

#define RmiOpaque(page) ((RmiPageOpaque) PageGetSpecialPointer(page))

// Leaf and overflow pages hold two parallel arrays: keys first (aligned, so
// binary search reads them directly), then 6-byte TIDs. 14 bytes per entry.
#define RMI_ENTRY_SIZE (sizeof(int64) + sizeof(ItemPointerData))
#define RMI_CONTENT_SIZE \
	(BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(RmiPageOpaqueData)))
#define RMI_LEAF_CAPACITY ((int) (RMI_CONTENT_SIZE / RMI_ENTRY_SIZE))

#define RmiKeys(page) ((int64 *) PageGetContents(page))
#define RmiTids(page) \
	((ItemPointerData *) (PageGetContents(page) + RMI_LEAF_CAPACITY * sizeof(int64)))

typedef struct RmiLinear
{
	double		slope;
	double		intercept;
	int32		min_err;		// worst (actual - predicted), <= 0
	int32		max_err;		// worst (actual - predicted), >= 0
} RmiLinear;

typedef struct RmiKnot
{
	double		x;
	double		y;
	double		slope;
} RmiKnot;

typedef enum RmiStage1Kind
{
	RMI_S1_LINEAR,
	RMI_S1_LOGLINEAR,
	RMI_S1_PIECEWISE
} RmiStage1Kind;

// Written to block 0. Never changes after build.
typedef struct RmiMetaData
{
	uint32		magic;
	uint32		version;
	int64		nentries;		// at build
	int64		min_key;		// keys are stored to the model as key - min_key
	uint32		nmodels;
	uint32		nknots;
	uint32		stage1_kind;
	uint32		per_leaf;
	double		s1_slope;
	double		s1_intercept;
	BlockNumber first_model_blk;
	BlockNumber first_leaf_blk;
	BlockNumber nleaves;
	double		avg_abs_err;
	double		avg_window;
	int64		max_window;
} RmiMetaData;

// Per-backend copy of the meta page and models, kept in rd_amcache. A REINDEX
// or TRUNCATE invalidates the relcache entry, which throws this away.
typedef struct RmiCache
{
	RmiMetaData meta;
	RmiKnot    *knots;
	RmiLinear  *models;
} RmiCache;

// An index entry held in memory during build.
typedef struct RmiEntry
{
	int64		key;
	ItemPointerData tid;
} RmiEntry;

typedef struct RmiOptions
{
	int32		vl_len_;
	int			fillfactor;
	int			models;			// 0 = choose from the number of rows
} RmiOptions;

#define RMI_DEFAULT_FILLFACTOR 90

// What the trainer produces. knots and models are palloc'd.
typedef struct RmiTrained
{
	RmiMetaData meta;
	RmiKnot    *knots;
	RmiLinear  *models;
} RmiTrained;

// rmi_model.c
extern void rmi_train(const RmiEntry *entries, int64 n, uint32 nmodels, RmiTrained *out);

// rmi_am.c
extern RmiCache *rmi_get_cache(Relation index);
extern bytea *rmi_options(Datum reloptions, bool validate);

// rmi_build.c
extern IndexBuildResult *rmi_build(Relation heap, Relation index, IndexInfo *indexInfo);
extern void rmi_buildempty(Relation index);

// rmi_scan.c
extern Buffer rmi_locate_leaf(Relation index, RmiCache *cache, int64 key, int lockmode,
							  BlockNumber *leaf_no);
extern IndexScanDesc rmi_beginscan(Relation index, int nkeys, int norderbys);
extern void rmi_rescan(IndexScanDesc scan, ScanKey keys, int nkeys,
					   ScanKey orderbys, int norderbys);
extern bool rmi_gettuple(IndexScanDesc scan, ScanDirection dir);
extern int64 rmi_getbitmap(IndexScanDesc scan, TIDBitmap *tbm);
extern void rmi_endscan(IndexScanDesc scan);

// rmi_insert.c
extern bool rmi_insert(Relation index, Datum *values, bool *isnull, ItemPointer ht_ctid,
					   Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
					   IndexInfo *indexInfo);
extern IndexBulkDeleteResult *rmi_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
											 IndexBulkDeleteCallback callback, void *cb_state);
extern IndexBulkDeleteResult *rmi_vacuumcleanup(IndexVacuumInfo *info,
												IndexBulkDeleteResult *stats);

// Index of the first key on the page that is >= key (may equal nentries).
static inline int
rmi_page_lower_bound(const int64 *keys, int n, int64 key)
{
	int			lo = 0;
	int			hi = n;

	while (lo < hi)
	{
		int			mid = lo + (hi - lo) / 2;

		if (keys[mid] < key)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

// Keys are measured from min_key, which handles negative int8 values and keeps
// precision. Keys below the minimum (only possible through inserts) clamp to 0.
static inline double
rmi_key_x(const RmiMetaData *meta, int64 key)
{
	if (key <= meta->min_key)
		return 0.0;
	return (double) ((uint64) key - (uint64) meta->min_key);
}

static inline double
rmi_clamp(double v, double lo, double hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

// Clamping to the segment's own y range keeps stage 1 monotone across knots.
static inline double
rmi_piecewise_eval(const RmiKnot *knots, uint32 nknots, double x)
{
	const RmiKnot *base = knots;
	uint32		n = nknots - 1;

	while (n > 1)
	{
		uint32		half = n >> 1;

		if (base[half].x <= x)
			base += half;
		n -= half;
	}
	return rmi_clamp(base->y + (x - base->x) * base->slope, base[0].y, base[1].y);
}

extern double rmi_log1p(double x);

// Stage 1: which stage-2 model owns this key.
static inline uint32
rmi_model_index(const RmiMetaData *meta, const RmiKnot *knots, double x)
{
	double		m;

	if (meta->stage1_kind == RMI_S1_PIECEWISE)
		m = rmi_piecewise_eval(knots, meta->nknots, x);
	else
	{
		if (meta->stage1_kind == RMI_S1_LOGLINEAR)
			x = rmi_log1p(x);
		m = meta->s1_slope * x + meta->s1_intercept;
	}
	return (uint32) rmi_clamp(m, 0.0, (double) (meta->nmodels - 1));
}

// Stage 2: predicted position in the sorted array as built.
static inline int64
rmi_predict_pos(const RmiMetaData *meta, const RmiLinear *model, double x)
{
	double		p = model->slope * x + model->intercept;

	return (int64) rmi_clamp(p, 0.0, (double) (meta->nentries - 1));
}

#endif							// RMI_H
