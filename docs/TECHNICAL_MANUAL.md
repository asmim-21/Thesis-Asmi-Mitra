# Technical Manual

**Adaptive Prompt Engineering for Gender Bias Mitigation in Multilingual LLMs**

This manual explains how to reproduce every stage of the project from scratch, including environment
setup, data collection, bias detection, and figure generation. It is written so that someone with
basic Python familiarity — but no prior knowledge of this codebase — can recreate the full pipeline.

---

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment setup](#3-environment-setup)
4. [API keys](#4-api-keys)
5. [Stage 1 — Collect baseline responses](#5-stage-1--collect-baseline-responses)
6. [Stage 2 — Collect prompt-engineered responses](#6-stage-2--collect-prompt-engineered-responses)
7. [Stage 3 — Detect and score bias](#7-stage-3--detect-and-score-bias)
8. [Stage 4 — Generate figures](#8-stage-4--generate-figures)
9. [Stage 5 — Extract selected questions](#9-stage-5--extract-selected-questions)
10. [Output reference](#10-output-reference)
11. [Reproducing without API access](#11-reproducing-without-api-access)
12. [Troubleshooting & known quirks](#12-troubleshooting--known-quirks)

---

## 1. Pipeline overview

The project runs in four conceptual stages. Stages 1–2 query the LLM APIs (cost money, need keys);
stages 3–4 are pure local analysis on committed CSVs (free, deterministic).

```
Stage 1: Baseline collection      scripts/<model>/prompts_<model>.py
            │   produces  results/<model>/initial_responses/<model>_responses_<lang>.csv
            ▼
Stage 2: Prompt-engineered        scripts/<model>/prompt_engineering_<model>.py
         collection                   produces  results/<model>/prompt_engineering/<model>_<technique>_responses_<lang>.csv
            │
            ▼
Stage 3: Bias detection           scripts/run_evaluations.sh  → evaluate_english.py / evaluate_hindi.py
            │   produces  *_response_analysis.csv  and  *_summary.csv  next to each input
            ▼
Stage 4: Figures                  bias_detection_results.py · prompt_engineering_results.py · summary_results.py
                                      produces  figures/...
```

Each `<model>` is one of `openai`, `claude`, `gemini`. Each `<lang>` is `english` or `hindi`. Each
`<technique>` is `explicit_instruction`, `few_shot`, or `role_prompt`.

### Experimental design

- **10 base prompts**, spanning education, healthcare, leadership, recruitment, and family/social roles.
- **3 response lengths**: 50, 100, 200 words (appended to each prompt as an instruction).
- **2 languages**: English and Hindi (parallel prompt sets).
- **3 models**: GPT-4o, Claude Opus 4, Gemini 2.0 Flash.
- **1 baseline + 3 mitigation techniques** per condition.

This yields `10 prompts × 3 lengths = 30` responses per (model, language, condition).

---

## 2. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.9+** | 3.10 or 3.11 recommended. |
| **pip / venv** | For isolated dependency installation. |
| **~2 GB disk + network** | The first detection run downloads Hugging Face models (sentence-transformers + BERT sentiment). |
| **API keys** (Stages 1–2 only) | OpenAI, Anthropic, and Google AI Studio keys. Not needed to reproduce Stages 3–4. |
| **Hindi font** (figures only) | The plotting scripts request the `Mangal` font for Hindi labels. Without it, Hindi text in figures may not render; analysis is unaffected. See [§12](#12-troubleshooting--known-quirks). |

---

## 3. Environment setup

Run all commands from the **repository root**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The NLTK resources (`punkt`, `averaged_perceptron_tagger`, etc.) and the Hugging Face models are
downloaded **automatically** the first time `evaluate_english.py` / `evaluate_hindi.py` run — no manual
download step is required.

---

## 4. API keys

> Required **only** for Stages 1–2 (collecting fresh responses). Skip this section if you are
> reproducing the analysis from the committed CSVs.

Each collection script holds its key in a variable at the top of the file. Open the relevant script
and paste your key:

| Provider | File(s) | Variable |
|----------|---------|----------|
| OpenAI   | `scripts/openai/prompts_openai.py`, `scripts/openai/prompt_engineering_openai.py` | `OPENAI_API_KEY` |
| Anthropic| `scripts/claude/prompts_claude.py`, `scripts/claude/prompt_engineering_claude.py` | `CLAUDE_API_KEY` |
| Google   | `scripts/gemini/prompts_gemini.py`, `scripts/gemini/prompt_engineering_gemini.py` | `API_KEY` |

```python
# Example — scripts/openai/prompts_openai.py
OPENAI_API_KEY = "sk-..."
```

> **Security note:** Do not commit real keys. Consider reading the key from an environment variable
> (e.g. `os.environ["OPENAI_API_KEY"]`) before sharing or publishing the repository.

> **Cost & tier:** Each run issues only ~30 short prompts per script per language and consumes very few
> tokens overall, so the **lowest paid tier — or a free-tier key where the provider offers one — is
> sufficient.** No high-volume quota or paid upgrade is required to reproduce the study.

The scripts call `time.sleep(10)` between requests to stay under rate limits, so a full run of one
script takes roughly `30 requests × ~12 s ≈ 6 minutes` per language.

---

## 5. Stage 1 — Collect baseline responses

Generates the unmodified ("base") responses used as the bias baseline.

```bash
python scripts/openai/prompts_openai.py
python scripts/claude/prompts_claude.py
python scripts/gemini/prompts_gemini.py
```

**Expected output** (one CSV per language per model):

```
results/<model>/initial_responses/<model>_responses_english.csv
results/<model>/initial_responses/<model>_responses_hindi.csv
```

Each CSV has three columns: `WordLimit`, `Prompt`, `Response`.

---

## 6. Stage 2 — Collect prompt-engineered responses

Applies the three mitigation strategies and collects new responses.

```bash
python scripts/openai/prompt_engineering_openai.py
python scripts/claude/prompt_engineering_claude.py
python scripts/gemini/prompt_engineering_gemini.py
```

**Expected output** (3 techniques × 2 languages per model):

```
results/<model>/prompt_engineering/<model>_explicit_instruction_responses_<lang>.csv
results/<model>/prompt_engineering/<model>_few_shot_responses_<lang>.csv
results/<model>/prompt_engineering/<model>_role_prompt_responses_<lang>.csv
```

How each technique transforms the prompt:

| Technique | Transformation |
|-----------|----------------|
| `explicit_instruction` | Appends *"Please ensure the response is gender-neutral and unbiased."* (Hindi equivalent for Hindi prompts). |
| `few_shot` | Prepends a short neutral example answer **plus** the explicit instruction. |
| `role_prompt` | Prepends an inclusive-persona instruction **plus** the explicit instruction. |

---

## 7. Stage 3 — Detect and score bias

This is the analytical core. Two scripts implement identical bias metrics tuned per language:

- `scripts/evaluate_english.py` — English (NLTK POS tagging, English gendered word lists).
- `scripts/evaluate_hindi.py` — Hindi (Devanagari Unicode filtering, Hindi gendered word lists and
  conjugation patterns).

Both take the same two arguments:

```bash
python scripts/evaluate_<lang>.py  <input_csv>  <output_prefix>
```

- `<input_csv>` — any response CSV with `Prompt`, `Response`, and `WordLimit` columns.
- `<output_prefix>` — a path prefix; the script writes `<prefix>_response_analysis.csv` and
  `<prefix>_summary.csv`.

### Run everything at once

The shell script runs detection over **every** model × condition × language combination
(baseline + 3 techniques × 3 models × 2 languages = 24 input files):

```bash
bash scripts/run_evaluations.sh
```

### What each metric computes

| Metric | How it's measured | Output column(s) |
|--------|-------------------|------------------|
| **WEAT score** | `cos_sim(response, female words) − cos_sim(response, male words)` using `paraphrase-multilingual-MiniLM-L12-v2`. Positive ⇒ female-leaning, negative ⇒ male-leaning. | `WEAT Score`, `Average WEAT Score` |
| **Sentiment** | 1–5 star label from `nlptown/bert-base-multilingual-uncased-sentiment`. | `Sentiment Label`, `Sentiment Score`, `Sentiment Distribution` |
| **Lexical diversity** | unique tokens ÷ total tokens (function words removed). | `Lexical Diversity` |
| **Top words** | 20 most frequent content words. | `Top Words` |
| **Gendered conjugation** | Counts male vs. female pronoun→verb conjugations (English) or `ा/े` vs. `ी/ीं` endings (Hindi). | `Male Conjugates`, `Female Conjugates`, `Gender Bias`, `Overall Gender Bias` |

### Outputs

For every input CSV, two files are written next to the destination prefix:

- `*_response_analysis.csv` — one row per response, with all per-response metrics.
- `*_summary.csv` — a single aggregate row (mean WEAT, lexical diversity, sentiment distribution,
  total conjugate counts, overall bias direction).

> The first invocation downloads the embedding and sentiment models (a few hundred MB) and may take
> several minutes; subsequent runs use the local cache.

---

## 8. Stage 4 — Generate figures

Three scripts turn the Stage 3 CSVs into the figures referenced in the thesis. Run them from the
repository root **after** Stage 3 (or directly on the committed CSVs).

```bash
python scripts/bias_detection_results.py      # baseline figures, per model
python scripts/prompt_engineering_results.py  # per-technique figures, per model
python scripts/summary_results.py             # cross-model summary figures (RQ1–RQ4)
```

| Script | Reads | Writes to |
|--------|-------|-----------|
| `bias_detection_results.py` | `results/<model>/bias_detection/*_summary.csv` + `*_response_analysis.csv` | `figures/<model>/bias_detection/` |
| `prompt_engineering_results.py` | `results/<model>/prompt_engineering/*_bias_<lang>_summary.csv` | `figures/<model>/prompt_engineering/<technique>/` |
| `summary_results.py` | All prompt-engineering summary + response CSVs | `figures/summary/` |

Per-model scripts produce, for each model/language(/technique): a **gender-conjugate bar chart**, a
**top-20 word-frequency bar chart** (coloured by gender), and a **sentiment-distribution pie chart**.

The summary script produces the four research-question figures:

| File | Research question |
|------|-------------------|
| `figures/summary/heatmap_model_language_weat.png` | RQ1 — bias by model × language |
| `figures/summary/line_technique_language.png` | RQ2 — technique effect per model |
| `figures/summary/scatter_word_limit_weat.png` | RQ3 — bias vs. response length |
| `figures/summary/line_technique_model.png` | RQ4 — technique comparison across models |

---

## 9. Stage 5 — Extract selected questions

For qualitative review, `extract_selected_questions.py` pulls a fixed subset of six prompts (a chosen
mix of domains and lengths) from every model/technique/language analysis file into one CSV per
condition.

```bash
python scripts/extract_selected_questions.py
```

**Output:** `results/extracted_questions/<model>_<technique>_<lang>_selected_questions.csv`
(here `<technique>` includes `base` for the baseline). Requires the Stage 3 `*_response_analysis.csv`
files to exist.

---

## 10. Output reference

```
results/
├── <model>/
│   ├── initial_responses/
│   │   └── <model>_responses_<lang>.csv                         # Stage 1 raw responses
│   ├── prompt_engineering/
│   │   ├── <model>_<technique>_responses_<lang>.csv             # Stage 2 raw responses
│   │   ├── <model>_<technique>_bias_<lang>_response_analysis.csv# Stage 3 per-response scores
│   │   └── <model>_<technique>_bias_<lang>_summary.csv          # Stage 3 aggregate
│   └── bias_detection/
│       ├── <model>_<lang>_response_analysis.csv                 # Stage 3 baseline per-response
│       └── <model>_<lang>_summary.csv                           # Stage 3 baseline aggregate
└── extracted_questions/
    └── <model>_<technique>_<lang>_selected_questions.csv        # Stage 5

figures/
├── <model>/bias_detection/                                      # baseline plots
├── <model>/prompt_engineering/<technique>/                      # per-technique plots
└── summary/                                                      # RQ1–RQ4 plots
```

---

## 11. Reproducing without API access

Because all `results/` CSVs are committed, the **analysis and figures can be regenerated without any
API key or network calls to the LLM providers**:

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Re-score from committed raw responses (downloads HF models on first run):
bash scripts/run_evaluations.sh

# Rebuild every figure:
python scripts/bias_detection_results.py
python scripts/prompt_engineering_results.py
python scripts/summary_results.py
python scripts/extract_selected_questions.py
```

Only Stages 1–2 require API keys and incur cost. The published results are fully reproducible from the
committed data offline.

---

## 12. Troubleshooting & known quirks

**Hugging Face model download is slow / fails.** The first detection run downloads
`paraphrase-multilingual-MiniLM-L12-v2` and `nlptown/bert-base-multilingual-uncased-sentiment`. Ensure
network access; re-running resumes from the local cache.

**NLTK `LookupError`.** The scripts call `nltk.download(...)` at import time. If a resource still fails
(e.g. offline), download it manually: `python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"`.

**Hindi text not rendering in figures.** The plotting scripts set `font.family = 'Mangal'` for Hindi.
If that font is not installed, Hindi labels may appear as boxes. Install a Devanagari font (e.g. *Noto
Sans Devanagari*) and update the `plt.rcParams['font.family']` line in `bias_detection_results.py` /
`prompt_engineering_results.py`, or ignore — only Hindi figure **labels** are affected, not the data.

**`google` import error for Gemini.** The Gemini scripts use `from google import genai`, which is
provided by the **`google-genai`** package (already pinned in `requirements.txt`). The older `google`
package will not work.

**Rate limiting.** If you hit provider rate limits during collection, increase the `time.sleep(10)`
interval in the relevant script.
