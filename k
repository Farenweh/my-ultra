#!/usr/bin/env bash

WHITELIST="\.vscode-server|nvitop|nputop|code-server|keep-alive"
KEYWORDS="pytorch|dataloader|python"
FORCE_KEYWORD="debug"

CURRENT_USER="$(whoami)"
SELF_PID="$$"

echo "正在扫描进程..."

PROC_SNAPSHOT="$(
    ps -u "$CURRENT_USER" -o pid=,ppid=,stat=,args=
)"

# 找出匹配规则但属于 zombie/defunct 的进程，仅提示，不 kill
# PPID=1 的 zombie 进程直接忽略
mapfile -t zombie_pids < <(
    awk -v self="$SELF_PID" \
        -v wl="$WHITELIST" \
        -v kw="$KEYWORDS" \
        -v force="$FORCE_KEYWORD" '
        $1 == self || $2 == self { next }

        # 忽略 PPID=1 的 zombie/defunct
        $3 ~ /Z/ && $2 == 1 { next }

        # 记录其他 zombie/defunct
        $3 ~ /Z/ {
            if ($0 ~ force || ($0 ~ kw && $0 !~ wl)) {
                print $1
            }
            next
        }
    ' <<< "$PROC_SNAPSHOT"
)

# 找出真正需要 kill 的非 zombie 进程
mapfile -t filtered_pids < <(
    awk -v self="$SELF_PID" \
        -v wl="$WHITELIST" \
        -v kw="$KEYWORDS" \
        -v force="$FORCE_KEYWORD" '
        $1 == self || $2 == self { next }

        # 忽略 PPID=1 的 zombie/defunct
        $3 ~ /Z/ && $2 == 1 { next }

        # 跳过其他 zombie/defunct
        $3 ~ /Z/ { next }

        # 包含 debug 的进程无视白名单
        $0 ~ force {
            print $1
            next
        }

        # 包含关键词且不在白名单
        $0 ~ kw && $0 !~ wl {
            print $1
        }
    ' <<< "$PROC_SNAPSHOT"
)

if [ "${#zombie_pids[@]}" -gt 0 ]; then
    echo
    echo "检测到以下匹配规则但无法直接 kill 的 zombie/defunct 进程："
    ps -o pid,ppid,stat,cmd -p "${zombie_pids[@]}"

    echo
    echo "说明：这些进程已经退出，只剩进程表记录。需要处理它们的父进程 PPID。"
    echo "注意：PPID=1 的 zombie/defunct 进程已被忽略。"
fi

if [ "${#filtered_pids[@]}" -eq 0 ]; then
    echo
    echo "没有找到需要终止的非 zombie 进程"
    exit 0
fi

echo
echo "以下非 zombie 进程将被终止（包含 '$FORCE_KEYWORD' 的进程无视白名单）："
ps -o pid,ppid,stat,cmd -p "${filtered_pids[@]}"

echo
echo "警告：这将强制终止以上进程（SIGKILL）！"
read -r -p "请输入 'yes' 确认执行，输入其他任何内容将退出: " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "操作已取消"
    exit 0
fi

echo "正在终止进程..."

for pid in "${filtered_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid"
        echo "已终止 PID: $pid"
    else
        echo "PID 已不存在，跳过: $pid"
    fi
done

echo "操作完成"

if [ "${#zombie_pids[@]}" -gt 0 ]; then
    echo
    echo "仍需关注的 zombie/defunct 进程父进程："

    zombie_ppids="$(
        ps -o ppid= -p "${zombie_pids[@]}" 2>/dev/null | awk '$1 != 1 {print $1}' | sort -u
    )"

    if [ -n "$zombie_ppids" ]; then
        # shellcheck disable=SC2086
        ps -o pid,ppid,stat,etime,cmd -p $zombie_ppids
    else
        echo "未找到需要关注的 zombie 父进程。PPID=1 的 zombie 已忽略。"
    fi
fi