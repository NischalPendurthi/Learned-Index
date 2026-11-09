// Trains the 2-stage RMI. Same method as src/rmi.cpp, with one change: rows can
// share a key here, so models are fit on the first position of each distinct
// key. That is the position a lookup has to find.
#include "rmi.h"

#include <math.h>

double
rmi_log1p(double x)
{
	return log(x + 1.0);
}

typedef struct Fit
{
	double		slope;
	double		intercept;
} Fit;

static inline long double
xform(const RmiMetaData *meta, int64 key, bool use_log)
{
	double		x = rmi_key_x(meta, key);

	return (long double) (use_log ? rmi_log1p(x) : x);
}

// Least squares over entries[lo, hi) with y = position.
static Fit
fit_range(const RmiMetaData *meta, const RmiEntry *e, int64 lo, int64 hi, bool use_log)
{
	Fit			fallback = {0.0, (double) lo};
	long double mx = 0.0L,
				my = 0.0L,
				sxx = 0.0L,
				sxy = 0.0L,
				slope,
				intercept;
	int64		cnt = 0;
	double		s,
				b;

	for (int64 i = lo; i < hi; i++)
	{
		if (i > lo && e[i].key == e[i - 1].key)
			continue;
		mx += xform(meta, e[i].key, use_log);
		my += (long double) i;
		cnt++;
	}
	if (cnt < 2)
		return fallback;
	mx /= cnt;
	my /= cnt;

	for (int64 i = lo; i < hi; i++)
	{
		long double dx,
					dy;

		if (i > lo && e[i].key == e[i - 1].key)
			continue;
		dx = xform(meta, e[i].key, use_log) - mx;
		dy = (long double) i - my;
		sxx += dx * dx;
		sxy += dx * dy;
	}
	if (sxx <= 0.0L)
		return fallback;

	slope = sxy / sxx;
	intercept = my - slope * mx;
	s = (double) slope;
	b = (double) intercept;
	if (!isfinite(s) || !isfinite(b) || s <= 0.0)
		return fallback;
	return (Fit) {s, b};
}

// Equal-depth knots through the CDF. Returns the number of knots, 0 if the keys
// don't give at least one segment.
static uint32
fit_piecewise(const RmiMetaData *meta, const RmiEntry *e, int64 n, uint32 nmodels, RmiKnot *knots)
{
	uint32		segments = Max(16, Min(nmodels / 4, 256));
	uint32		count = 0;

	for (uint32 s = 0; s <= segments; s++)
	{
		int64		pos = (s == segments) ? n - 1 : (int64) ((long double) s * n / segments);
		double		x;

		// a knot in the middle of a run of equal keys belongs at the run's start
		while (pos > 0 && e[pos].key == e[pos - 1].key)
			pos--;

		x = rmi_key_x(meta, e[pos].key);
		if (count > 0 && x <= knots[count - 1].x)
			continue;
		knots[count].x = x;
		knots[count].y = (double) ((long double) pos * nmodels / n);
		count++;
	}
	if (count < 2)
		return 0;

	knots[count - 1].y = (double) nmodels;
	for (uint32 i = 0; i + 1 < count; i++)
		knots[i].slope = (knots[i + 1].y - knots[i].y) / (knots[i + 1].x - knots[i].x);
	knots[count - 1].slope = 0.0;
	return count;
}

// Sum of (keys per model)^2 for the stage 1 currently set in meta.
static long double
stage1_cost(const RmiMetaData *meta, const RmiKnot *knots, const RmiEntry *e, int64 n,
			uint32 *counts)
{
	long double cost = 0.0L;

	memset(counts, 0, sizeof(uint32) * meta->nmodels);
	for (int64 i = 0; i < n; i++)
		counts[rmi_model_index(meta, knots, rmi_key_x(meta, e[i].key))]++;
	for (uint32 m = 0; m < meta->nmodels; m++)
		cost += (long double) counts[m] * (long double) counts[m];
	return cost;
}

void
rmi_train(const RmiEntry *e, int64 n, uint32 nmodels, RmiTrained *out)
{
	RmiMetaData *meta = &out->meta;
	double		scale = (double) nmodels / (double) n;
	uint32	   *counts;
	RmiKnot    *knots;
	uint32		nknots;
	Fit			line,
				logline;
	long double best_cost = -1.0L;
	RmiStage1Kind best_kind = RMI_S1_LINEAR;
	double		best_slope = 0.0,
				best_intercept = 0.0;
	long double total_err = 0.0L,
				total_window = 0.0L;
	int64		distinct = 0,
				max_window = 0;
	int64		i;
	uint32		next_model = 0;

	meta->nentries = n;
	meta->min_key = e[0].key;
	meta->nmodels = nmodels;
	meta->nknots = 0;

	counts = palloc(sizeof(uint32) * nmodels);
	knots = palloc(sizeof(RmiKnot) * 257);
	out->models = palloc0(sizeof(RmiLinear) * nmodels);

	line = fit_range(meta, e, 0, n, false);
	logline = fit_range(meta, e, 0, n, true);
	nknots = fit_piecewise(meta, e, n, nmodels, knots);

	for (int c = 0; c < 3; c++)
	{
		long double cost;

		if (c < 2)
		{
			Fit			f = (c == 0) ? line : logline;

			if (!(f.slope > 0.0))
				continue;
			meta->stage1_kind = (c == 0) ? RMI_S1_LINEAR : RMI_S1_LOGLINEAR;
			meta->s1_slope = f.slope * scale;
			meta->s1_intercept = f.intercept * scale;
		}
		else
		{
			if (nknots == 0)
				continue;
			meta->stage1_kind = RMI_S1_PIECEWISE;
			meta->nknots = nknots;
		}

		cost = stage1_cost(meta, knots, e, n, counts);

		// piecewise costs a knot search per lookup, so it must clearly win
		if (best_cost < 0.0L || cost < best_cost * (c == 2 ? 0.95L : 1.0L))
		{
			best_cost = cost;
			best_kind = meta->stage1_kind;
			best_slope = meta->s1_slope;
			best_intercept = meta->s1_intercept;
		}
	}

	if (best_cost < 0.0L)
	{
		// one distinct key, or all fits failed: send everything to model 0
		best_kind = RMI_S1_LINEAR;
		best_slope = 0.0;
		best_intercept = 0.0;
	}

	meta->stage1_kind = best_kind;
	meta->s1_slope = best_slope;
	meta->s1_intercept = best_intercept;
	meta->nknots = (best_kind == RMI_S1_PIECEWISE) ? nknots : 0;
	out->knots = knots;

	// Stage 2. Stage 1 is monotone, so each model owns a contiguous run.
	i = 0;
	while (i < n)
	{
		uint32		m = rmi_model_index(meta, knots, rmi_key_x(meta, e[i].key));
		int64		lo = i;
		int64		hi = i;
		RmiLinear  *lm = &out->models[m];
		Fit			f;
		int64		mn = PG_INT64_MAX;
		int64		mx = PG_INT64_MIN;
		int64		run_distinct = 0;

		while (hi < n && rmi_model_index(meta, knots, rmi_key_x(meta, e[hi].key)) == m)
			hi++;

		// skipped models hold no keys; absent keys routed there belong here
		for (; next_model < m; next_model++)
			out->models[next_model].intercept = (double) lo;
		next_model = m + 1;

		f = fit_range(meta, e, lo, hi, false);
		lm->slope = f.slope;
		lm->intercept = f.intercept;

		for (int64 j = lo; j < hi; j++)
		{
			int64		err;

			if (j > lo && e[j].key == e[j - 1].key)
				continue;
			err = j - rmi_predict_pos(meta, lm, rmi_key_x(meta, e[j].key));
			if (err < mn)
				mn = err;
			if (err > mx)
				mx = err;
			total_err += (long double) (err < 0 ? -err : err);
			run_distinct++;
		}

		// Bounds only narrow the page search, and the leaf fences correct any
		// miss, so clamping an enormous error here cannot lose a key.
		lm->min_err = (int32) Max(mn, PG_INT32_MIN);
		lm->max_err = (int32) Min(mx, PG_INT32_MAX);

		distinct += run_distinct;
		total_window += (long double) (mx - mn + 1) * run_distinct;
		if (mx - mn + 1 > max_window)
			max_window = mx - mn + 1;

		i = hi;
	}

	// models past the last key only see absent keys, which belong at the end
	for (; next_model < nmodels; next_model++)
		out->models[next_model].intercept = (double) (n - 1);

	meta->avg_abs_err = distinct ? (double) (total_err / distinct) : 0.0;
	meta->avg_window = distinct ? (double) (total_window / distinct) : 0.0;
	meta->max_window = max_window;

	pfree(counts);
}
