#!/usr/bin/env bash
# WITHIN: one command from public data to real results + within.live.html. Run from the within/ folder.
#
#   bash run_all.sh                 # PhysioNet + Tufts (automatic) + REFED (only if you have Hugging Face access)
#   NO_REFED=1 bash run_all.sh      # skip REFED entirely: no Hugging Face account needed
#   TUFTS=/path/to/size_30sec_150ts_stride_03ts bash run_all.sh   # reuse an existing Tufts folder
#   FULL=1 bash run_all.sh          # also train the 1D CNN (slow on CPU; installs PyTorch)
#
# REFED is gated: to include it, accept its licence on huggingface.co and run `huggingface-cli login` first.
# Without access, REFED is skipped and everything else still runs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/pipeline"
OUT=./out; mkdir -p "$OUT"
log() { echo; echo "=== $* ==="; }
pipi() { [ "${SKIP_INSTALL:-0}" = "1" ] && return 0; python -m pip install -q "$@" 2>/dev/null || python -m pip install -q --break-system-packages "$@"; }
if [ "${FULL:-0}" = "1" ]; then pipi -r requirements.txt huggingface_hub matplotlib; MODELS=svm,xgb,cnn
else pipi -r requirements-fast.txt huggingface_hub matplotlib; MODELS=svm,xgb; fi
ARGS=()

# 1. PhysioNet mental-fnirs: open access, ~100 MB, fNIRS + TCD during the same n-back task
log "1/6 PhysioNet: download"
PN=physionet.org/files/mental-fnirs/1.0
if [ ! -d "$PN/NIRS_data" ]; then
  command -v wget >/dev/null || { echo "wget is needed (Colab has it; on a Mac: brew install wget)"; exit 1; }
  wget -q -r -N -c -np https://physionet.org/files/mental-fnirs/1.0/
fi

# 2. Evidence: TCD blood-flow velocity (real, from the authors' per-session summary)
log "2/6 Evidence: TCD blood-flow velocity"
(cd "$ROOT/evidence" && python compute_tcd.py --csv "$ROOT/pipeline/$PN/TCD_data/Summary-statistics_MCA.csv" > /dev/null) \
  && echo "evidence/physionet_tcd_results.json + chart updated"
python physionet_adapter.py --root "$PN" --out "$OUT"          # adds the fNIRS side from raw intensities
ARGS+=(--physionet-out "$OUT")

# 3. Tufts fNIRS2MW: 68 participants, public Box folder, ~2.3 GB (THINK)
log "3/6 Tufts: download + cognitive-load models"
if [ -z "${TUFTS:-}" ]; then
  TUFTS="$(pwd)/tufts_30s"
  set +e; python tufts_download.py --out "$TUFTS" --min-subjects "${MIN_TUFTS:-20}"; TDL=$?; set -e
else
  TDL=0
fi
N_TUFTS=$(find "$TUFTS" -maxdepth 1 -name 'sub_*.csv' 2>/dev/null | wc -l | tr -d ' ')
if [ "$TDL" -eq 0 ] && [ "$N_TUFTS" -ge "${MIN_TUFTS:-20}" ]; then
  echo "Tufts participants available: $N_TUFTS"
  python tufts_train.py --windows "$TUFTS" --partition-plan partition_plan.txt --out "$OUT" --models "${MODELS/svm,/}"
  ARGS+=(--tufts-windows "$TUFTS" --tufts-out "$OUT")
else
  echo "Only $N_TUFTS Tufts participants available (fewer than ${MIN_TUFTS:-20}): skipping THINK; the site keeps its preview."
  TUFTS=""
fi

# 4. REFED (FEEL): only if Hugging Face access works; otherwise skipped
log "4/6 REFED: emotion models (fNIRS only)"
REFED=""
if [ "${NO_REFED:-0}" != "1" ]; then
  set +e
  REFED=$(python - <<'PY' 2>/dev/null
import glob, os
from huggingface_hub import snapshot_download
d = os.path.abspath("../REFED-dataset")
find = lambda: glob.glob(os.path.join(d, "**", "fNIRS_videos.mat"), recursive=True)
if not find():
    snapshot_download("REFED2025/REFED-dataset", repo_type="dataset", local_dir=d,
                      allow_patterns=["*fNIRS_videos.mat", "*fNIRS_baselines.mat", "*annotations*", "*label*.mat", "*.md"])
if not find():   # layout differs from the official code: fall back to the full download
    snapshot_download("REFED2025/REFED-dataset", repo_type="dataset", local_dir=d)
print(os.path.dirname(os.path.dirname(os.path.dirname(find()[0]))))
PY
  ); [ $? -eq 0 ] || REFED=""
  set -e
fi
if [ -n "$REFED" ]; then
  echo "REFED root: $REFED"
  python refed_train.py --root "$REFED" --out "$OUT" --models "$MODELS" --video-holdout-models xgb
  ARGS+=(--refed-root "$REFED" --refed-out "$OUT")
else
  echo "REFED skipped (no Hugging Face access, or NO_REFED=1): FEEL keeps the published benchmark."
fi

# 5. Personalization: does a person's own labelled data help?
log "5/6 Personalization experiment"
PZ=()
[ -n "$REFED" ] && PZ+=(--refed-features "$OUT/refed_features.npz")
[ -n "$TUFTS" ] && PZ+=(--tufts-windows "$TUFTS" --partition-plan partition_plan.txt)
if [ ${#PZ[@]} -gt 0 ]; then
  python personalize.py "${PZ[@]}" --out "$OUT"; ARGS+=(--personalize-out "$OUT")
else
  echo "No REFED or Tufts data: skipped."
fi

# 6. Real results into the site, plain-language summary, one zip
log "6/6 Building within.live.html"
python export_demo.py "${ARGS[@]}" --template ../within.html --html-out ../within.live.html
python summarize_results.py --out "$OUT" | tee "$OUT/SUMMARY.txt"
cd "$ROOT" && rm -f within_results.zip && zip -qj within_results.zip pipeline/out/*.json pipeline/out/SUMMARY.txt \
  within.live.html evidence/physionet_tcd_results.json evidence/physionet_tcd_chart.png
echo
echo "Done. Results: within_results.zip   Site with real numbers: within.live.html"
