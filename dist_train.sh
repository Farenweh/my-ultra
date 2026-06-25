#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH_CONFIG="${SSH_CONFIG:-/root/.ssh/config}"

discover_remote_worker_host() {
  local config="${1:-}"
  [[ -r "$config" ]] || return 1

  awk '
    tolower($1) == "host" {
      for (i = 2; i <= NF; i++) {
        if ($i ~ /-worker-/) {
          print $i
          exit
        }
      }
    }
  ' "$config"
}

resolve_ssh_hostname() {
  local host="$1"
  local ssh_args=()

  if [[ -r "$SSH_CONFIG" ]]; then
    ssh_args+=(-F "$SSH_CONFIG")
  fi

  ssh "${ssh_args[@]}" -G "$host" 2>/dev/null | awk 'tolower($1) == "hostname" {print $2; exit}'
}

resolve_host_ip() {
  local host="$1"
  getent hosts "$host" 2>/dev/null | awk '{print $1; exit}'
}

sanitize_log_token() {
  local value="$1"
  printf '%s' "$value" | tr -c 'A-Za-z0-9_.-' '_'
}

# 默认从机用户和目标，可以通过环境变量覆盖 REMOTE_HOST。
REMOTE_USER="${REMOTE_USER:-root}"

if [[ -z "${REMOTE_HOST:-}" ]]; then
  REMOTE_HOST="$(discover_remote_worker_host "$SSH_CONFIG" || true)"
  if [[ -z "$REMOTE_HOST" ]]; then
    echo "无法自动发现远端 worker。请设置 REMOTE_HOST，或确认 ${SSH_CONFIG} 中存在 Host *-worker-*。" >&2
    exit 1
  fi
fi

REMOTE_SSH_HOSTNAME="$(resolve_ssh_hostname "$REMOTE_HOST" || true)"
REMOTE_HOST_IP="$(resolve_host_ip "${REMOTE_SSH_HOSTNAME:-$REMOTE_HOST}" || true)"
REMOTE_HOST_IP="${REMOTE_HOST_IP:-$REMOTE_HOST}"
REMOTE_LOG_HOST="$(sanitize_log_token "$REMOTE_HOST_IP")"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
# 修改这里以更改ssh密钥路径，可以通过环境变量覆盖
SSH_KEY="${SSH_KEY:-${SCRIPT_DIR}/key}"

LOCAL_REPO_DIR="${LOCAL_REPO_DIR:-${SCRIPT_DIR}}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-${LOCAL_REPO_DIR}}"
ENV_SCRIPT="${ENV_SCRIPT:-/root/work/filestorage/zhoufei/hermite/deps/bashrc_env.sh}"
MAMBA_ENV="${MAMBA_ENV:-torch}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train.py}"
SYNC_ENV="${SYNC_ENV:-1}"
SYNC_ENV_EXCLUDE="${SYNC_ENV_EXCLUDE:-}"
REMOTE_USE_TINI="${REMOTE_USE_TINI:-1}"
REMOTE_TINI_BIN="${REMOTE_TINI_BIN:-/usr/bin/tini}"

DEFAULT_MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
MASTER_ADDR="${MASTER_ADDR:-${DEFAULT_MASTER_ADDR}}"
MASTER_PORT="${MASTER_PORT:-29500}"
DIST_WORLD_SIZE="${DIST_WORLD_SIZE:-3}"
REMOTE_CHECK_TIMEOUT="${REMOTE_CHECK_TIMEOUT:-10}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/runs/dist_train}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
REMOTE_LOG="${LOG_DIR}/remote-${REMOTE_LOG_HOST}-${RUN_ID}.log"

mkdir -p "$LOG_DIR"
chmod 600 "$SSH_KEY"

if [[ -z "$MASTER_ADDR" ]]; then
  echo "无法确定 MASTER_ADDR。请显式设置 MASTER_ADDR 或确保 hostname -I 返回一个 IP。" >&2
  exit 1
fi

SSH_OPTS=(
)

if [[ -r "$SSH_CONFIG" ]]; then
  SSH_OPTS+=(-F "$SSH_CONFIG")
fi

SSH_OPTS+=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -i "$SSH_KEY"
)

remote_pid=""
local_pid=""

cleanup() {
  local status=$?
  if [[ -n "${local_pid}" ]] && kill -0 "$local_pid" 2>/dev/null; then
    echo "正在停止本地训练启动器..."
    kill "$local_pid" 2>/dev/null || true
    wait "$local_pid" 2>/dev/null || true
  fi
  if [[ -n "${remote_pid}" ]] && kill -0 "$remote_pid" 2>/dev/null; then
    echo "正在停止 ${REMOTE} 上的远程训练启动器..."
    kill "$remote_pid" 2>/dev/null || true
    wait "$remote_pid" 2>/dev/null || true
  fi
  exit "$status"
}

trap cleanup EXIT INT TERM

check_remote_connectivity() {
  echo "检查与 ${REMOTE} 的 SSH 连接和清空远程 Python 进程..."
  if ! timeout "$REMOTE_CHECK_TIMEOUT" ssh "${SSH_OPTS[@]}" "$REMOTE" "bash -s -- $(printf '%q' "$REMOTE_REPO_DIR") $(printf '%q' "$REMOTE_USE_TINI") $(printf '%q' "$REMOTE_TINI_BIN")" <<'REMOTE_PREFLIGHT'
set -Eeuo pipefail

REMOTE_REPO_DIR="$1"
REMOTE_USE_TINI="$2"
REMOTE_TINI_BIN="$3"

if [[ "${REMOTE_USE_TINI}" != "0" && "${REMOTE_USE_TINI,,}" != "false" ]] && [[ ! -x "$REMOTE_TINI_BIN" ]]; then
  echo "远端 tini 不可用: ${REMOTE_TINI_BIN}" >&2
  exit 1
fi

cd "$REMOTE_REPO_DIR"
printf 'yes\n' | bash k
REMOTE_PREFLIGHT
  then
    echo "无法到达 ${REMOTE} 通过 SSH 或在训练前清除远程 Python 进程。" >&2
    echo "SSH key: ${SSH_KEY}" >&2
    echo "Timeout: ${REMOTE_CHECK_TIMEOUT}s" >&2
    echo "Remote repo: ${REMOTE_REPO_DIR}" >&2
    exit 1
  fi
}

build_synced_env_payload() {
  if [[ "${SYNC_ENV}" == "0" || "${SYNC_ENV,,}" == "false" ]]; then
    printf ''
    return
  fi

  python3 - <<'PY'
import base64
import os
import re
import sys

skip = {
    # Distributed launch values are node-specific and are set explicitly by this script.
    "K8S_TRAINING",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    # Launcher internals and local shell/SSH process state should not override remote arguments.
    "REMOTE_USER",
    "REMOTE_HOST",
    "REMOTE_HOST_IP",
    "REMOTE_SSH_HOSTNAME",
    "REMOTE_LOG_HOST",
    "REMOTE",
    "SSH_KEY",
    "SSH_CONFIG",
    "LOCAL_REPO_DIR",
    "REMOTE_REPO_DIR",
    "ENV_SCRIPT",
    "MAMBA_ENV",
    "TRAIN_ENTRYPOINT",
    "SYNC_ENV",
    "SYNC_ENV_EXCLUDE",
    "REMOTE_USE_TINI",
    "REMOTE_TINI_BIN",
    "LOG_DIR",
    "RUN_ID",
    "REMOTE_LOG",
    "DIST_WORLD_SIZE",
    "REMOTE_CHECK_TIMEOUT",
    "PWD",
    "OLDPWD",
    "SHLVL",
    "_",
    "SSH_AUTH_SOCK",
    "SSH_CLIENT",
    "SSH_CONNECTION",
    "SSH_TTY",
}
extra_excludes = os.environ.get("SYNC_ENV_EXCLUDE", "")
for name in re.split(r"[\s,:]+", extra_excludes):
    if name:
        skip.add(name)

valid_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
items = []
for key_b, value_b in os.environb.items():
    key = key_b.decode("utf-8", "surrogateescape")
    if key in skip or key.startswith("BASH_FUNC_") or not valid_name.match(key):
        continue
    items.append(key_b + b"=" + value_b)

if items:
    sys.stdout.write(base64.b64encode(b"\0".join(items) + b"\0").decode())
PY
}

build_remote_command() {
  local command="bash -s --"
  local arg

  if [[ "${REMOTE_USE_TINI}" != "0" && "${REMOTE_USE_TINI,,}" != "false" ]]; then
    command="$(printf '%q' "$REMOTE_TINI_BIN") -s -- bash -s --"
  fi

  for arg in "$REMOTE_REPO_DIR" "$ENV_SCRIPT" "$MAMBA_ENV" "$TRAIN_ENTRYPOINT" "$MASTER_ADDR" "$MASTER_PORT" "$DIST_WORLD_SIZE" "$@"; do
    command+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$command"
}

run_remote() {
  local remote_command
  local synced_env_payload
  remote_command="$(build_remote_command "$@")"
  synced_env_payload="$(build_synced_env_payload)"
  {
    printf 'SYNCED_ENV_PAYLOAD=%q\n' "$synced_env_payload"
    cat <<'REMOTE_SCRIPT'
set -Eeuo pipefail

REMOTE_REPO_DIR="$1"
ENV_SCRIPT="$2"
MAMBA_ENV="$3"
TRAIN_ENTRYPOINT="$4"
MASTER_ADDR="$5"
MASTER_PORT="$6"
WORLD_SIZE="$7"
shift 7

import_synced_env() {
  local payload="$1"
  [[ -n "$payload" ]] || return 0

  local entry key
  while IFS= read -r -d '' entry; do
    [[ "$entry" == *=* ]] || continue
    key="${entry%%=*}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    case "$key" in
      K8S_TRAINING|MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|REMOTE_REPO_DIR|ENV_SCRIPT|MAMBA_ENV|TRAIN_ENTRYPOINT)
        continue
        ;;
    esac
    export "$entry"
  done < <(printf '%s' "$payload" | base64 -d)
}

import_synced_env "$SYNCED_ENV_PAYLOAD"

export K8S_TRAINING=1
export MASTER_ADDR
export MASTER_PORT
export RANK=1
export WORLD_SIZE
unset LOCAL_RANK

source "$ENV_SCRIPT"
mamba activate "$MAMBA_ENV"

cd "$REMOTE_REPO_DIR"
exec python "$TRAIN_ENTRYPOINT" "$@"
REMOTE_SCRIPT
  } | ssh "${SSH_OPTS[@]}" "$REMOTE" "$remote_command"
}

run_local() {
  export K8S_TRAINING=1
  export MASTER_ADDR
  export MASTER_PORT
  export RANK=0
  export WORLD_SIZE="$DIST_WORLD_SIZE"
  unset LOCAL_RANK

  source "$ENV_SCRIPT"
  mamba activate "$MAMBA_ENV"

  cd "$LOCAL_REPO_DIR"
  python "$TRAIN_ENTRYPOINT" "$@"
}

print_remote_log_tail() {
  if [[ -s "$REMOTE_LOG" ]]; then
    echo "远端日志最后 80 行："
    tail -80 "$REMOTE_LOG" || true
  fi
}

check_remote_connectivity

echo "远端 worker: ${REMOTE}，解析地址: ${REMOTE_HOST_IP}"
echo "在 ${REMOTE} 上启动远程训练启动器，远端控制台输出日志将写入 ${REMOTE_LOG}..."
run_remote "$@" >"$REMOTE_LOG" 2>&1 &
remote_pid=$!

echo "训练入口: ${TRAIN_ENTRYPOINT}"
echo "以参数MASTER_ADDR=${MASTER_ADDR}、MASTER_PORT=${MASTER_PORT}、WORLD_SIZE=${DIST_WORLD_SIZE}启动本地RANK 0"
local_status=0
remote_status=0
first_status=0
finished_pid=""

run_local "$@" &
local_pid=$!

if wait -n -p finished_pid "$local_pid" "$remote_pid"; then
  first_status=0
else
  first_status=$?
fi

if [[ "$finished_pid" == "$local_pid" ]]; then
  local_status=$first_status
  local_pid=""
  if (( local_status != 0 )); then
    echo "本地训练启动器提前退出，正在停止远程训练启动器..."
    kill "$remote_pid" 2>/dev/null || true
  fi
  wait "$remote_pid" || remote_status=$?
  remote_pid=""
elif [[ "$finished_pid" == "$remote_pid" ]]; then
  remote_status=$first_status
  remote_pid=""
  if (( remote_status != 0 )); then
    echo "远程训练启动器提前退出，正在停止本地训练启动器..."
    print_remote_log_tail
    kill "$local_pid" 2>/dev/null || true
  fi
  wait "$local_pid" || local_status=$?
  local_pid=""
else
  echo "无法识别已结束的训练启动器：${finished_pid}" >&2
  kill "$local_pid" "$remote_pid" 2>/dev/null || true
  wait "$local_pid" 2>/dev/null || local_status=$?
  wait "$remote_pid" 2>/dev/null || remote_status=$?
  local_pid=""
  remote_pid=""
fi

if (( local_status != 0 || remote_status != 0 )); then
  echo "dist_train.sh 失败：本地状态=${local_status}，远程状态=${remote_status}"
  echo "Remote log: ${REMOTE_LOG}"
  print_remote_log_tail
  exit 1
fi

echo "dist_train.sh 已成功完成"
