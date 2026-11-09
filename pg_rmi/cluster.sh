#!/usr/bin/env bash
# A private PostgreSQL 18 cluster for the rmi index. It lives outside the repo,
# needs no sudo, and leaves the system cluster alone.
#
#   ./cluster.sh init      create the cluster and start it
#   ./cluster.sh install   build the extension and install it for this cluster
#   ./cluster.sh start | stop | restart | crash | psql [args]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RMI_CLUSTER_ROOT:-$HERE/../../pg_rmi_cluster}"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
DATA="$ROOT/data"
SOCK="$ROOT/sock"
EXT="$ROOT/ext"
LOG="$ROOT/server.log"
PORT=5544
BIN=/usr/lib/postgresql/18/bin

run_pg_ctl() { "$BIN/pg_ctl" -D "$DATA" -l "$LOG" -w "$@"; }

case "${1:-}" in
  init)
    mkdir -p "$SOCK"
    "$BIN/initdb" -D "$DATA" -U postgres -A trust --no-sync >/dev/null
    cat >> "$DATA/postgresql.conf" <<EOF
port = $PORT
listen_addresses = ''
unix_socket_directories = '$SOCK'
shared_buffers = 1GB
max_wal_size = 8GB
checkpoint_timeout = 30min
dynamic_library_path = '$EXT/usr/lib/postgresql/18/lib:\$libdir'
extension_control_path = '$EXT/usr/share/postgresql/18:\$system'
EOF
    run_pg_ctl start >/dev/null
    "$BIN/createdb" -h "$SOCK" -p $PORT -U postgres rmi_test
    echo "cluster ready at $DATA, port $PORT"
    ;;
  install)
    make -C "$HERE" -s PG_CONFIG=/usr/bin/pg_config
    make -C "$HERE" -s PG_CONFIG=/usr/bin/pg_config install DESTDIR="$EXT"
    ;;
  start) run_pg_ctl start >/dev/null ;;
  stop) run_pg_ctl stop -m fast >/dev/null ;;
  restart) run_pg_ctl restart -m fast >/dev/null ;;
  crash) run_pg_ctl stop -m immediate >/dev/null ;;
  psql) shift; exec "$BIN/psql" -h "$SOCK" -p $PORT -U postgres -X -d "${RMI_DB:-rmi_test}" "$@" ;;
  env) echo "PGHOST=$SOCK PGPORT=$PORT PGUSER=postgres PGDATABASE=${RMI_DB:-rmi_test} DATA=$DATA BIN=$BIN" ;;
  *) echo "usage: $0 init|install|start|stop|restart|crash|psql|env" >&2; exit 2 ;;
esac
