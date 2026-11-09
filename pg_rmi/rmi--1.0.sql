\echo Use "CREATE EXTENSION rmi" to load this file. \quit

CREATE FUNCTION rmi_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE ACCESS METHOD rmi TYPE INDEX HANDLER rmi_handler;

CREATE OPERATOR FAMILY rmi_integer_ops USING rmi;

CREATE OPERATOR CLASS rmi_int8_ops
    DEFAULT FOR TYPE int8 USING rmi FAMILY rmi_integer_ops AS
    OPERATOR 1 <  (int8, int8),
    OPERATOR 2 <= (int8, int8),
    OPERATOR 3 =  (int8, int8),
    OPERATOR 4 >= (int8, int8),
    OPERATOR 5 >  (int8, int8),
    FUNCTION 1 btint8cmp(int8, int8);

-- Without these, WHERE k = 5 (an int4 literal) could not use the index.
ALTER OPERATOR FAMILY rmi_integer_ops USING rmi ADD
    OPERATOR 1 <  (int8, int4),
    OPERATOR 2 <= (int8, int4),
    OPERATOR 3 =  (int8, int4),
    OPERATOR 4 >= (int8, int4),
    OPERATOR 5 >  (int8, int4),
    FUNCTION 1 (int8, int4) btint84cmp(int8, int4),
    OPERATOR 1 <  (int8, int2),
    OPERATOR 2 <= (int8, int2),
    OPERATOR 3 =  (int8, int2),
    OPERATOR 4 >= (int8, int2),
    OPERATOR 5 >  (int8, int2),
    FUNCTION 1 (int8, int2) btint82cmp(int8, int2);

CREATE FUNCTION rmi_stats(
    index regclass,
    OUT entries_at_build bigint,
    OUT models int,
    OUT stage1 text,
    OUT knots int,
    OUT avg_abs_err float8,
    OUT avg_window float8,
    OUT max_window bigint,
    OUT leaf_pages bigint,
    OUT leaf_entries bigint,
    OUT overflow_pages bigint,
    OUT overflow_entries bigint,
    OUT overflow_pages_per_leaf float8
) AS 'MODULE_PATHNAME' LANGUAGE C STRICT;

REVOKE ALL ON FUNCTION rmi_stats(regclass) FROM PUBLIC;
