# Evidence

Three kinds of numbers appear in WITHIN. Only the first two may be stated as fact today.

## 1. Computed by this project from real public data

**PhysioNet TCD workload result** (`physionet_tcd_results.json`, `physionet_tcd_per_participant.csv`, `physionet_tcd_chart.png`)

Source: Mukli, Yabluchanskiy & Csipo (2021), PhysioNet mental-fnirs v1.0, doi:10.13026/zfb2-1g43, file `TCD_data/Summary-statistics_MCA.csv`. Measure: % change in mean middle-cerebral-artery blood-flow velocity (left/right average), each n-back block vs the 0-back block before it.

| Task | Mean change ± SE | Median | Participants up | One-sided Wilcoxon p |
|---|---|---|---|---|
| 1-back | +0.63% ± 0.68 | +0.93% | 8 of 14 | 0.29 (not significant) |
| 2-back | **+4.79% ± 1.23** | +3.00% | **13 of 14** | **0.00012** |

The 2-back increase is also larger than the 1-back increase (p = 0.00012). Reproduce: `python compute_tcd.py` (downloads the summary file, no account needed).

Safe wording: "In public data, cerebral blood-flow velocity rose 4.8% on average during the hardest memory task, in 13 of 14 participants." Limits: 14 people, one lab, one session each; TCD is not Temple's Flow.

## 2. Published by others (quote with citation; not our results)

- **REFED** (Ning et al., NeurIPS 2025 Datasets & Benchmarks), fNIRS only, three classes, within-subject leave-one-trial-out: MDNet 60.53% valence / 66.47% arousal; SVM 57.30% / 64.28%. Chance with three balanced classes: 33%.
- **Tufts fNIRS2MW** (Huang, Wang et al., NeurIPS 2021 Datasets & Benchmarks): 68 participants, forehead fNIRS at 5.2 Hz, four n-back levels.

## 3. Pending: produced only by running the pipeline

- FEEL: our cross-subject REFED results, including the people-and-videos-held-out check
- THINK: our cross-subject Tufts results (all 68 participants download automatically)
- PhysioNet fNIRS side (prefrontal HbO), computed from raw intensities
- Personalization: whether a person's own labels give a reliable gain

Run `run_on_colab.ipynb` (only manual step: accept REFED's licence on Hugging Face and paste a read token). It returns `within_results.zip`. Until then, the site shows these as "PENDING" or "SIMULATED — NOT A RESEARCH RESULT".
