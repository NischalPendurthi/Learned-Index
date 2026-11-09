MODULE_big = rmi
OBJS = rmi_am.o rmi_build.o rmi_insert.o rmi_model.o rmi_scan.o
EXTENSION = rmi
DATA = rmi--1.0.sql
SHLIB_LINK = -lm
PG_CFLAGS = -Wall -Wno-unused-parameter

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
