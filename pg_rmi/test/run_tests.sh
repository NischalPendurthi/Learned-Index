#!/usr/bin/env bash
# Correctness suite for the rmi index. Every check compares against a seq scan.
# Stops at the first failure.  Usage: test/run_tests.sh [rows] [rounds]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CL="$HERE/../cluster.sh"
ROWS="${1:-200000}"
ROUNDS="${2:-60}"
Q() { "$CL" psql -v ON_ERROR_STOP=1 -qAt "$@"; }

Q -c 'DROP EXTENSION IF EXISTS rmi CASCADE' -c 'CREATE EXTENSION rmi' 2>/dev/null
Q -f "$HERE/checks.sql"

echo "== build and read"
Q <<SQL
DROP TABLE IF EXISTS t_dense, t_log, t_uni, t_neg, t_dup, t_null, t_empty, t_one, t_same;
CREATE TABLE t_dense AS SELECT i::bigint AS k FROM generate_series(1, $ROWS) i ORDER BY random();
CREATE TABLE t_log AS SELECT (exp(random_normal(0, 2)) * 1e9)::bigint AS k FROM generate_series(1, $ROWS);
CREATE TABLE t_uni AS SELECT (random() * 9e18)::bigint AS k FROM generate_series(1, $ROWS);
CREATE TABLE t_neg AS SELECT ((random() - 0.5) * 1.8e19)::bigint AS k FROM generate_series(1, $ROWS);
-- runs of equal keys long enough to cross many leaf pages
CREATE TABLE t_dup AS SELECT (floor(power(random(), 4) * 50))::bigint * 1000 AS k FROM generate_series(1, $ROWS);
CREATE TABLE t_null AS SELECT CASE WHEN i % 3 = 0 THEN NULL ELSE i::bigint END AS k FROM generate_series(1, $ROWS) i;
CREATE TABLE t_empty (k bigint);
CREATE TABLE t_one AS SELECT 42::bigint AS k;
CREATE TABLE t_same AS SELECT 7::bigint AS k FROM generate_series(1, 5000);
ALTER TABLE t_dense SET (autovacuum_enabled = off);
CREATE INDEX i_dense ON t_dense USING rmi (k);
CREATE INDEX i_log ON t_log USING rmi (k);
CREATE INDEX i_uni ON t_uni USING rmi (k);
CREATE INDEX i_neg ON t_neg USING rmi (k);
CREATE INDEX i_dup ON t_dup USING rmi (k);
CREATE INDEX i_null ON t_null USING rmi (k);
CREATE INDEX i_empty ON t_empty USING rmi (k);
CREATE INDEX i_one ON t_one USING rmi (k);
CREATE INDEX i_same ON t_same USING rmi (k);
ANALYZE;
SQL

TABLES="t_dense t_log t_uni t_neg t_dup t_null t_empty t_one t_same"
check_all() { for t in $TABLES; do Q -c "SELECT rmi_check('$t', $ROUNDS)"; done; }
check_all

echo "== model quality"
Q -F ' ' -c "SELECT i, models, stage1, round(avg_window::numeric,1), max_window, leaf_pages
             FROM unnest(ARRAY['i_dense','i_log','i_uni','i_neg','i_dup']) i, rmi_stats(i::regclass)"

echo "== inserts (new min and max keys, into empty tables, enough to overflow leaves)"
Q <<SQL
INSERT INTO t_empty SELECT (random() * 1e6)::bigint FROM generate_series(1, 20000);
INSERT INTO t_one SELECT (random() * 1e6)::bigint FROM generate_series(1, 20000);
INSERT INTO t_same SELECT 7 FROM generate_series(1, 5000);
INSERT INTO t_same SELECT (random() * 20)::bigint FROM generate_series(1, 5000);
INSERT INTO t_dense SELECT (random() * $ROWS * 2)::bigint - $ROWS / 2 FROM generate_series(1, $ROWS);
INSERT INTO t_log SELECT (exp(random_normal(0, 2)) * 1e9)::bigint FROM generate_series(1, $ROWS / 2);
INSERT INTO t_dup SELECT (floor(random() * 50))::bigint * 1000 FROM generate_series(1, $ROWS / 2);
INSERT INTO t_neg VALUES (9223372036854775807), (-9223372036854775808), (NULL);
ANALYZE;
SQL
check_all
Q -F ' ' -c "SELECT i, leaf_entries, overflow_pages, overflow_entries
             FROM unnest(ARRAY['i_dense','i_log','i_dup','i_empty','i_same']) i, rmi_stats(i::regclass)"

echo "== delete, update, vacuum"
Q <<SQL
DELETE FROM t_dense WHERE k % 3 = 0;
DELETE FROM t_dup WHERE k = 0;
UPDATE t_log SET k = k + 1 WHERE k % 7 = 0;
VACUUM t_dense, t_dup, t_log;
ANALYZE;
SQL
check_all

echo "== reindex retrains and removes overflow"
Q -c "REINDEX TABLE t_dense" -c "REINDEX TABLE t_dup" -c "REINDEX TABLE t_log" -c "ANALYZE"
check_all
Q -F ' ' -c "SELECT i, leaf_entries, overflow_pages FROM unnest(ARRAY['i_dense','i_dup','i_log']) i, rmi_stats(i::regclass)"

echo "== concurrent readers and writers"
ENV="$("$CL" env)"
eval "export $ENV"
PGB="$BIN/pgbench"
SCRIPT="$(mktemp -d)"
cat > "$SCRIPT/write.sql" <<SQL
\set k random(-1000000, 3000000)
INSERT INTO t_dense VALUES (:k);
SQL
cat > "$SCRIPT/read.sql" <<SQL
\set k random(-1000000, 3000000)
SELECT count(*) FROM t_dense WHERE k = :k;
SELECT count(*) FROM t_dense WHERE k BETWEEN :k AND :k + 50;
SQL
cat > "$SCRIPT/del.sql" <<SQL
\set k random(-1000000, 3000000)
DELETE FROM t_dense WHERE k = :k;
SQL
OUT="$("$PGB" -n -c 16 -j 4 -T 20 -f "$SCRIPT/write.sql@5" -f "$SCRIPT/read.sql@4" -f "$SCRIPT/del.sql@1" 2>&1)" || { echo "$OUT"; exit 1; }
echo "$OUT" | grep -E 'number of transactions actually|number of failed|tps'
echo "$OUT" | grep -q 'number of failed transactions: 0 ' || { echo "$OUT"; exit 1; }
Q -c "VACUUM t_dense" -c "ANALYZE t_dense"
Q -c "SELECT rmi_check('t_dense', $ROUNDS)"

echo "== crash recovery (immediate stop right after unflushed inserts)"
Q <<SQL
CREATE TABLE IF NOT EXISTS crash_marker (n bigint);
TRUNCATE crash_marker;
INSERT INTO t_dense SELECT (random() * 4e6)::bigint - 1000000 FROM generate_series(1, 200000);
INSERT INTO crash_marker SELECT count(*) FROM t_dense;
SQL
BEFORE="$(Q -c 'SELECT n FROM crash_marker')"
"$CL" crash
"$CL" start
AFTER="$(Q -c 'SELECT count(*) FROM t_dense')"
[ "$BEFORE" = "$AFTER" ] || { echo "row count changed across crash: $BEFORE vs $AFTER"; exit 1; }
Q -c "ANALYZE t_dense"
Q -c "SELECT rmi_check('t_dense', $ROUNDS)"
echo "recovered $AFTER rows"

rm -rf "$SCRIPT"
echo "ALL CHECKS PASSED"
