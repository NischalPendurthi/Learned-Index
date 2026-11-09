-- rmi_check(table, n): run n random predicates through the rmi index, as both a
-- plain index scan and a bitmap scan, and compare the exact set of rows with a
-- sequential scan. Raises an exception on the first difference.
CREATE OR REPLACE FUNCTION rmi_row_set(tbl regclass, pred text) RETURNS text AS $$
DECLARE r text;
BEGIN
    EXECUTE format('SELECT count(*) || '':'' || coalesce(md5(string_agg(ctid::text, '','' ORDER BY ctid)), '''')
                    FROM %s WHERE %s', tbl, pred) INTO r;
    RETURN r;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION rmi_plan_uses_index(tbl regclass, pred text) RETURNS boolean AS $$
DECLARE line text; found boolean := false;
BEGIN
    FOR line IN EXECUTE format('EXPLAIN SELECT ctid FROM %s WHERE %s', tbl, pred) LOOP
        IF line LIKE '%Index Scan using%' OR line LIKE '%Bitmap Index Scan on%' THEN
            found := true;
        END IF;
    END LOOP;
    RETURN found;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION rmi_check_pred(tbl regclass, pred text) RETURNS void AS $$
DECLARE expected text; got text;
BEGIN
    PERFORM set_config('enable_seqscan', 'on', true);
    PERFORM set_config('enable_indexscan', 'off', true);
    PERFORM set_config('enable_bitmapscan', 'off', true);
    expected := rmi_row_set(tbl, pred);

    PERFORM set_config('enable_seqscan', 'off', true);
    PERFORM set_config('enable_indexscan', 'on', true);
    IF NOT rmi_plan_uses_index(tbl, pred) THEN
        RAISE EXCEPTION 'index not used by index scan plan: % on %', pred, tbl;
    END IF;
    got := rmi_row_set(tbl, pred);
    IF got IS DISTINCT FROM expected THEN
        RAISE EXCEPTION 'index scan differs on % WHERE %: expected %, got %', tbl, pred, expected, got;
    END IF;

    PERFORM set_config('enable_indexscan', 'off', true);
    PERFORM set_config('enable_bitmapscan', 'on', true);
    IF NOT rmi_plan_uses_index(tbl, pred) THEN
        RAISE EXCEPTION 'index not used by bitmap plan: % on %', pred, tbl;
    END IF;
    got := rmi_row_set(tbl, pred);
    IF got IS DISTINCT FROM expected THEN
        RAISE EXCEPTION 'bitmap scan differs on % WHERE %: expected %, got %', tbl, pred, expected, got;
    END IF;

    PERFORM set_config('enable_indexscan', 'on', true);
    PERFORM set_config('enable_seqscan', 'on', true);
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION rmi_check(tbl regclass, n int) RETURNS text AS $$
DECLARE
    keys bigint[];
    a bigint; b bigint; tmp bigint;
    cnt int := 0;
    nkeys int;
BEGIN
    EXECUTE format('SELECT array_agg(k) FROM (SELECT k FROM %s WHERE k IS NOT NULL ORDER BY random() LIMIT 2000) s', tbl)
        INTO keys;

    -- the extremes of bigint, on any table
    PERFORM rmi_check_pred(tbl, 'k = 9223372036854775807');
    PERFORM rmi_check_pred(tbl, 'k = -9223372036854775808');
    PERFORM rmi_check_pred(tbl, 'k > 9223372036854775807');
    PERFORM rmi_check_pred(tbl, 'k < -9223372036854775808');
    PERFORM rmi_check_pred(tbl, 'k = 5');
    PERFORM rmi_check_pred(tbl, 'k = 5::int2');
    PERFORM rmi_check_pred(tbl, 'k >= 5 AND k < 500::int4');

    IF keys IS NULL THEN
        RETURN format('%s: ok (no keys, extremes only)', tbl);
    END IF;
    nkeys := array_length(keys, 1);

    FOR i IN 1..n LOOP
        a := keys[1 + floor(random() * nkeys)::int];
        b := keys[1 + floor(random() * nkeys)::int];
        IF a > b THEN tmp := a; a := b; b := tmp; END IF;

        -- present keys, and neighbours that are usually absent
        PERFORM rmi_check_pred(tbl, format('k = %s', a));
        IF a > -9223372036854775808 THEN PERFORM rmi_check_pred(tbl, format('k = %s', a - 1)); END IF;
        IF a < 9223372036854775807 THEN PERFORM rmi_check_pred(tbl, format('k = %s', a + 1)); END IF;

        CASE i % 6
            WHEN 0 THEN PERFORM rmi_check_pred(tbl, format('k < %s', a));
            WHEN 1 THEN PERFORM rmi_check_pred(tbl, format('k <= %s', a));
            WHEN 2 THEN PERFORM rmi_check_pred(tbl, format('k > %s', b));
            WHEN 3 THEN PERFORM rmi_check_pred(tbl, format('k >= %s', b));
            WHEN 4 THEN PERFORM rmi_check_pred(tbl, format('k BETWEEN %s AND %s', a, b));
            ELSE
                -- range starting at an absent key, the case model bounds alone get wrong
                IF a < 9223372036854775807 THEN
                    PERFORM rmi_check_pred(tbl, format('k > %s AND k <= %s', a, b));
                END IF;
        END CASE;
        cnt := cnt + 1;
    END LOOP;
    RETURN format('%s: ok (%s rounds)', tbl, cnt);
END $$ LANGUAGE plpgsql;
