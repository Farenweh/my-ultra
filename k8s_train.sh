#!/usr/bin/env bash
set -e

export K8S_TRAINING=1
# shellcheck disable=SC1091
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/work/filestorage/zhoufei/hermite/deps/bashrc_env.sh
mamba activate torch

cd "/root/work/filestorage/zhoufei/hermite/my-ultra"
exec python train.py "$@"
