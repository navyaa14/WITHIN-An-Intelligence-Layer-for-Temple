# WITHIN

A product concept for Temple: what else could cerebral blood-flow signals reveal? This folder is the working prototype: the site, the evaluation pipeline, and the evidence. Independent work on public fNIRS/TCD data; not affiliated with Temple, no Temple data, no medical claims.

## Contents

| Path | What it is |
|---|---|
| `within.html` | The prototype site (one file). Model sections show labelled previews until real results are injected. |
| `run_all.sh` | One command: downloads data, trains, evaluates, builds `within.live.html` and `within_results.zip`. |
| `run_on_colab.ipynb` | The same, in Google Colab (no installs, free CPU). |
| `evidence/` | Real result available now, with the script that reproduces it. |
| `film/WITHIN_60s.mp4` | The 60-second concept film (voiced). Device: concept rendering; screens: illustrative. |
| `.gitignore` | Keeps datasets and generated outputs out of Git. |
| `pipeline/` | Training, evaluation and export code. |

## Real result already in this folder

PhysioNet mental-fnirs v1.0 (Mukli, Yabluchanskiy & Csipo, 2021; doi:10.13026/zfb2-1g43), middle-cerebral-artery blood-flow velocity (TCD), each n-back block vs the 0-back before it:

- **2-back: +4.79% ± 1.23 (mean ± SE), higher in 13 of 14 participants, one-sided Wilcoxon p = 0.00012**
- 1-back: +0.63% ± 0.68, 8 of 14, p = 0.29 (not significant)

Files: `evidence/physionet_tcd_results.json`, `physionet_tcd_per_participant.csv`, `physionet_tcd_chart.png`. Reproduce: `cd evidence && python compute_tcd.py`. TCD measures velocity in a large artery; it is not Temple's Flow.

## Run it

**Colab (easiest):** open `run_on_colab.ipynb` in Google Colab, upload this zip when asked, run the cells. Download `within_results.zip` at the end.

**Your own machine** (Python 3.10+, `wget`, `zip`):

```bash
bash run_all.sh                 # PhysioNet + Tufts automatic; REFED included only if you have Hugging Face access
NO_REFED=1 bash run_all.sh      # skip REFED: no account needed
TUFTS=/path/to/size_30sec_150ts_stride_03ts bash run_all.sh   # reuse a local Tufts copy
FULL=1 bash run_all.sh          # also train the 1D CNN (slow on CPU; installs PyTorch)
```

What it does, in order:

1. **PhysioNet** (open, ~100 MB): downloads fNIRS + TCD recordings.
2. **Evidence**: recomputes the TCD result above; `physionet_adapter.py` adds prefrontal HbO from raw fNIRS.
3. **Tufts fNIRS2MW** (68 participants, ~2.3 GB, public Box folder): cognitive-load models (THINK), official subject buckets.
4. **REFED** (optional, gated): emotion models (FEEL), fNIRS only. Without Hugging Face access this step is skipped and FEEL keeps the published benchmark.
5. **Personalization**: does a person's own labelled data improve accuracy? Uses whatever of REFED/Tufts is available.
6. **Export**: writes `within.live.html` (the site with real numbers), `pipeline/out/SUMMARY.txt` (plain-language results) and `within_results.zip`.

Outputs go to `pipeline/out/`. Only numbers printed in `SUMMARY.txt` should be quoted.

## Evaluation rules (enforced in code)

- **Subject-independent:** splits are grouped by participant; the run fails if anyone is in two of train/validation/test.
- **Headline metric:** balanced accuracy and macro-F1, next to per-person chance and a shuffled-label baseline.
- **Stimulus check (REFED):** every participant watched the same 15 videos, so a second test holds out people *and* videos together.
- **Personalization:** fixed per-person test half; a gain counts only if Holm-corrected Wilcoxon p < 0.05, the 95% bootstrap CI excludes zero, and the personalised model beats chance.
- **Tufts windows:** 30-second windows that span two workload levels are dropped before training.
- **REFED files:** labels are read from `annotations/<id>_label.mat` (Hugging Face layout) or `s<id>_label.mat`. The export checks the channel order (HbT ≈ HbO + HbR, HbO varies more than HbR) and warns if it fails; only HbT's position is confirmed by the official code.
- **Calibration:** each person's signals are scaled with their own unlabelled recordings (a short calibration period in a product).
- **Analysis assumptions (PhysioNet fNIRS):** Prahl extinction coefficients, DPF 6, 3 cm separation, 0.005–0.1 Hz band-pass, prefrontal channels 1–24. Results in SD units.

## Datasets and licences

- PhysioNet mental-fnirs v1.0: Mukli, Yabluchanskiy & Csipo (2021), doi:10.13026/zfb2-1g43. PhysioNet Contributor Review Health Data License.
- Tufts fNIRS2MW: Huang, Wang et al., NeurIPS 2021 Datasets & Benchmarks. CC BY 4.0.
- REFED: Ning et al., NeurIPS 2025 Datasets & Benchmarks. CC BY-NC-SA 4.0 (non-commercial), gated on Hugging Face.

Published benchmark quoted on the site (not our result): REFED fNIRS-only, three classes, within-subject: MDNet 60.53% valence / 66.47% arousal; SVM 57.30% / 64.28%.

## Status of results

This is the reproducible source package. The only computed result included is the PhysioNet TCD evidence. FEEL, THINK and personalization numbers appear only after `run_all.sh` (or the Colab notebook) has run; until then the site stays in labelled preview mode. Don't quote model accuracy you haven't generated.

## Before publishing

`LINKS.live` is set to the published prototype (turn on sharing on its page). Set `LINKS.github` near the top of the script in `within.html` to show the GitHub button.
