#!/usr/bin/env bash
# Run 3: upstream DeepFilterNet3 recipe, our licence-clean data.
#
# Prepares the data if needed, writes the run config once, then trains. Rerun
# the same command to resume: df.train continues from the latest checkpoint, and
# the config is never rewritten under an existing run.
#
#   wsl -d Ubuntu -u root -- bash /mnt/e/roommute/scripts/training/train.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../wsl_env.sh"
DATA="${DFN_DATA:-/root/dfn-data}"
RUN="${DFN_RUN:-/root/dfn-run3}"

bash "$HERE/prepare.sh"
[ -e "$RUN/config.ini" ] || python3 "$HERE/make_cfg.py" "$DATA/hdf5" "$RUN"

export DF_NO_PIN_MEMORY=1  # WSL cannot allocate pinned host memory
echo "### training -> $RUN  $(date '+%F %T')"
exec /root/dfn-rocm/bin/python -m df.train "$RUN/dataset.cfg" "$DATA/hdf5" "$RUN"
