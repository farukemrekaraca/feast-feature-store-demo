#!/usr/bin/env bash
# Live demo runner: press Enter to move to the next step.
# Usage: ./demo.sh          (interactive, for the presentation)
#        ./demo.sh --no-wait (run everything straight through)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore

WAIT=true
[[ "${1:-}" == "--no-wait" ]] && WAIT=false

step() {
  echo
  echo -e "\033[1;35m━━━ $1 ━━━\033[0m"
  if $WAIT; then read -r -p "(enter)" _; fi
}

step "0 · Generate two weeks of delivery data"
rm -rf feature_repo/data models  # clean slate: no registry, empty online store
python scripts/00_generate_data.py

step "1 · feast apply: register entities, feature views, feature service"
(cd feature_repo && feast apply 2>&1 | grep -v -E "RuntimeWarning|warnings.warn|^$")

step "2 · Training data: naive join vs Feast point-in-time join"
python scripts/01_point_in_time.py

step "3 · Train the model on both training sets"
python scripts/02_train.py

step "4 · Materialize to the online store and score live orders"
python scripts/03_materialize_and_serve.py

step "5 · Push a fresh value (streaming) and score again"
python scripts/04_push_fresh_features.py

echo
echo "Done. Extras: 'cd feature_repo && feast ui' (http://localhost:8888)"
echo "             'cd feature_repo && feast serve' + curl (see README)"
