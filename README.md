# WITHIN
### An Intelligence Layer for Temple

> **I didn’t build another health dashboard. I built the intelligence layer I think cerebral wearables are missing.**

**WITHIN** explores a simple question:

### What if Temple could go from showing that your physiology changed to helping you understand *what kind of moment it was*?

Temple already measures **Flow** — a real-time proxy for cerebral blood-flow dynamics.

WITHIN proposes a layer on top of that signal:

**Flow → interpretation → context → personalization**

It introduces three product primitives:

| Layer | What it asks | Product output |
|---|---|---|
| **FEEL** | *What kind of response is happening?* | Arousal + valence |
| **THINK** | *How much sustained mental demand is present?* | Cognitive-load state + recovery context |
| **MOMENTS** | *What was actually happening when physiology changed?* | User-labelled context → personal physiological model |

---

## The Idea

Most wearables stop at measurement.

They tell you that something changed.

WITHIN asks whether a cerebral wearable could eventually help answer:

- Was this response **high or low arousal**?
- Was it **positive or negative in valence**?
- Has **cognitive demand stayed unusually elevated**?
- What was happening when the signal changed?
- Does this physiological pattern mean something different **for this person**?

The product loop is:

```text
TEMPLE FLOW
     ↓
   WITHIN
     ↓
 ┌───────────────┐
 │ FEEL          │ → arousal + valence
 │ THINK         │ → cognitive load
 └───────────────┘
     ↓
   MOMENTS
     ↓
user-labelled context
     ↓
PERSONAL MODEL
     ↓
more useful interpretation over time
```

The core philosophy:

> **Your physiology notices the moment. You say what it was. WITHIN learns your pattern.**

---

# 01 — FEEL

## Understand the type of response

High physiological activation does not automatically mean stress.

Excitement, anticipation, frustration and stress can all involve elevated arousal.

FEEL separates the interpretation into two dimensions:

```text
Arousal → How activated is the response?
Valence → Is the response trending positive or negative?
```

Instead of:

```text
"You are stressed."
```

the product can represent something closer to:

```text
High arousal
Positive valence

Often associated with:
excitement · engagement · anticipation
```

The goal is **not emotion diagnosis**.

The goal is a more useful representation of physiological state.

### Research path

FEEL uses the public **REFED** dataset for the experimental pipeline:

- 32 participants
- synchronized EEG + fNIRS
- continuous valence/arousal annotations
- fNIRS-only path used by WITHIN
- subject-independent evaluation
- additional participant + stimulus holdout test

Until this repository's pipeline is run, the UI keeps FEEL model outputs explicitly marked as preview/pending.

---

# 02 — THINK

## Understand sustained mental demand

A single difficult moment is normal.

What becomes more useful is knowing when mental demand stays elevated for a long period.

THINK explores whether cerebral physiology can support a product representation such as:

```text
COGNITIVE LOAD

72
HIGH

Elevated for 46 min
```

followed by an actionable nudge:

```text
Mental demand has remained elevated.
Consider a short recovery period.
```

The value is not the number itself.

The value is turning a physiological pattern into something a person can **understand and act on**.

---

# 03 — MOMENTS

## Physiology detects the change. You add the context.

A population model can only go so far.

Two people can show similar physiological responses for completely different reasons.

So WITHIN closes the loop.

When a meaningful physiological change occurs:

```text
Your physiology changed.

What was happening?

[ Stress ]
[ Excitement ]
[ Focus ]
[ Frustration ]
[ Other ]
```

That label becomes personal context.

Over time:

```text
population model
      +
your labelled moments
      +
your physiological baseline
      ↓
personal model
```

The long-term product hypothesis is that **context supplied by the user can make interpretation more useful than physiology alone**.

And importantly, the repository tests that hypothesis instead of simply assuming personalization helps.

---

# From Measurement to Meaning

```text
                    ┌──────────────┐
                    │ TEMPLE FLOW  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    WITHIN    │
                    └──────┬───────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          ┌───────────┐         ┌───────────┐
          │   FEEL    │         │   THINK   │
          │           │         │           │
          │ Arousal   │         │ Cognitive │
          │ Valence   │         │   Load    │
          └─────┬─────┘         └─────┬─────┘
                └──────────┬───────────┘
                           ▼
                     ┌───────────┐
                     │  MOMENTS  │
                     └─────┬─────┘
                           ▼
                    User Context
                           │
                           ▼
                    Personal Model
```

**State → Context → Personalization**

That is the intelligence layer.

---

# What I Actually Built

This repository is more than a product mockup.

It contains the product experience, research evidence, evaluation pipeline and reproducible experiment framework behind the concept.

### Product

- Temple-style interactive product experience
- FEEL interface
- THINK interface
- MOMENTS timeline
- personal-model flow
- research/evidence views
- explicit preview vs real-result states
- 60-second voiced concept film

### ML / Evaluation

- REFED fNIRS emotion pipeline
- Tufts fNIRS2MW workload pipeline
- PhysioNet fNIRS/TCD adapter
- SVM and XGBoost baselines
- optional 1D CNN
- participant-independent splitting
- shuffled-label baseline
- balanced accuracy
- macro-F1
- participant-level chance
- stimulus-holdout evaluation
- personalization experiment
- bootstrap confidence intervals
- Holm-corrected statistical testing
- automatic result injection into the product experience

---

# One Real Result Already Reproduced

Before asking whether THINK could work on a cerebral wearable, I wanted to test the most basic premise:

### Does cerebral blood-flow velocity change as cognitive workload becomes harder?

Using the public **PhysioNet mental-fnirs v1.0** dataset, the repository reproduces the TCD workload analysis.

### Hardest workload condition — 2-back

**Average cerebral blood-flow velocity change**

# **+4.79%**

**13 / 14 participants increased**

**one-sided Wilcoxon p = 0.00012**

For comparison:

| Task | Mean change ± SE | Participants ↑ | p-value |
|---|---:|---:|---:|
| 1-back | +0.63% ± 0.68 | 8 / 14 | 0.29 |
| **2-back** | **+4.79% ± 1.23** | **13 / 14** | **0.00012** |

Reproduce it with:

```bash
cd evidence
python compute_tcd.py
```

Outputs:

```text
evidence/
├── physionet_tcd_results.json
├── physionet_tcd_per_participant.csv
└── physionet_tcd_chart.png
```

### Important

This is **TCD middle-cerebral-artery blood-flow velocity**.

It is **not Temple Flow**.

The result supports investigating the hypothesis on actual Temple hardware. It does not validate WITHIN on Temple.

---

# Evaluation Designed to Be Hard to Fool

A nice-looking accuracy number was not enough.

The evaluation framework deliberately makes the models earn their results.

### 1. Participant-independent testing

Entire people are held out.

```text
TRAIN PEOPLE
     ≠
VALIDATION PEOPLE
     ≠
TEST PEOPLE
```

The pipeline fails if a participant appears in more than one split.

This prevents nearby windows from the same person's recording from leaking across train and test sets.

---

### 2. Balanced accuracy + macro-F1

Headline metrics are reported alongside:

- participant-level chance
- shuffled-label baseline

So performance is compared against something meaningful rather than an arbitrary percentage.

---

### 3. REFED participant + video holdout

REFED participants all view the same emotional stimuli.

A model could potentially learn properties of the videos rather than emotion-related physiology.

So WITHIN includes a stricter evaluation where:

```text
test participant = unseen
AND
test video = unseen
```

---

### 4. Clean Tufts workload windows

Thirty-second windows spanning more than one workload level are **removed before training**.

No mixed-label boundary windows are used as clean examples.

---

### 5. Personalization must prove itself

WITHIN does not assume personalization helps.

For a held-out individual:

```text
population model
       ↓
+ 1 labelled moment
+ 2 labelled moments
+ 4 labelled moments
+ 8 labelled moments
       ↓
same untouched personal test set
```

A gain is considered reliable only when:

- Holm-corrected Wilcoxon **p < 0.05**
- 95% bootstrap confidence interval excludes zero
- personalized performance remains above chance

If that does not happen, the system should say so.

---

# Datasets

### REFED — FEEL

**Ning et al., NeurIPS 2025 Datasets & Benchmarks**

- 32 participants
- 15 emotion-inducing trials per participant
- synchronized EEG + fNIRS
- continuous valence/arousal annotation
- fNIRS-only path used here
- CC BY-NC-SA 4.0
- gated through Hugging Face

Published fNIRS-only benchmark quoted in the prototype:

| Model | Valence | Arousal |
|---|---:|---:|
| MDNet | 60.53% | 66.47% |
| SVM | 57.30% | 64.28% |

Three-class chance ≈ 33%.

**These are published REFED results — not results produced by this repository.**

---

### Tufts fNIRS2MW — THINK

**Huang, Wang et al., NeurIPS 2021 Datasets & Benchmarks**

- 68 eligible participants
- forehead fNIRS
- 5.2 Hz
- 0-back / 1-back / 2-back / 3-back
- public 30-second windows
- CC BY 4.0

The downloader in this repository covers all **68 eligible participant files**.

---

### PhysioNet mental-fnirs — physiological bridge

**Mukli, Yabluchanskiy & Csipo, 2021**

Used for:

- TCD cerebral blood-flow velocity analysis
- raw fNIRS workload analysis
- prefrontal HbO processing

The TCD result above is the real computed result currently included in the repository.

---

# Run the Entire Experiment

## Google Colab — easiest

Open:

```text
run_on_colab.ipynb
```

Run the notebook from top to bottom.

REFED is the only gated dataset. If you have access, authenticate when prompted.

The pipeline handles the remaining workflow.

At the end it creates:

```text
within_results.zip
```

containing the generated results and updated experience.

---

## Local

Requirements:

```text
Python 3.10+
wget
zip
```

Run everything:

```bash
bash run_all.sh
```

Skip gated REFED:

```bash
NO_REFED=1 bash run_all.sh
```

Reuse an existing Tufts dataset:

```bash
TUFTS=/path/to/size_30sec_150ts_stride_03ts bash run_all.sh
```

Include the optional 1D CNN:

```bash
FULL=1 bash run_all.sh
```

---

# What the Pipeline Produces

```text
Public neurophysiology datasets
             │
             ▼
       preprocessing
             │
             ▼
 participant-level splits
             │
             ▼
     SVM / XGBoost / CNN
             │
             ▼
     leakage checks
             │
             ▼
 balanced accuracy + macro-F1
             │
             ├──────────────► shuffled-label baseline
             │
             ├──────────────► stimulus holdout
             │
             └──────────────► personalization test
             │
             ▼
       SUMMARY.txt
             │
             ▼
     within.live.html
```

Only results written into `SUMMARY.txt` should be quoted as results from this repository.

---

# Repository Structure

```text
WITHIN/
│
├── within.html
│   └── interactive product prototype
│
├── film/
│   └── WITHIN_60s.mp4
│       60-second voiced concept film
│
├── run_all.sh
│   └── end-to-end experiment runner
│
├── run_on_colab.ipynb
│   └── one-run Google Colab workflow
│
├── evidence/
│   ├── EVIDENCE.md
│   ├── compute_tcd.py
│   ├── physionet_tcd_results.json
│   ├── physionet_tcd_per_participant.csv
│   └── physionet_tcd_chart.png
│
└── pipeline/
    ├── refed_train.py
    ├── tufts_train.py
    ├── tufts_download.py
    ├── physionet_adapter.py
    ├── personalize.py
    ├── export_demo.py
    ├── summarize_results.py
    ├── common.py
    └── nets.py
```

---


