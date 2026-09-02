# Predicting Search Visibility Loss — Went-Dark Prediction Model

## What this is

A machine learning project that predicts whether a webpage will **go "dark"** — receive zero organic search clicks — the month after it was performing normally. It uses only information available up to that point (no future data), and ranks which February signals matter most for predicting a March drop-off.

**Who it's for:** SEO teams, content teams, or anyone managing a large number of pages who wants an early-warning signal for pages at risk of losing search visibility, so they can be reviewed before traffic is actually lost.

**Live paper:** https://prithvirajr203-pixel.github.io/Paper-Asisgnemnt-Flyrank-/

---

## What you'll find in this repo

| Path | What it is |
|---|---|
| `work/notebooks/capstone.ipynb` | The full, runnable notebook — data connection → features → models → results → recommendations |
| `submission/paper_url.txt` | Link to the deployed research paper |
| `docs/paper/index.html` | The deployed paper source |

---

## Setup (a stranger can follow this)

1. **Get access to the dataset** (free, instant approval):
   - Go to the `FlyRank/internship-warehouse` dataset on Hugging Face
   - Request access and accept the data-use terms — approval is instant
   - Get a Hugging Face access token from your HF account settings

2. **Open the notebook in Google Colab** (no local install needed):
   - Click the "Open in Colab" badge at the top of `work/notebooks/capstone.ipynb`, or open it directly at:
     `https://colab.research.google.com/github/prithvirajr203-pixel/Notebook-FlyRank/blob/main/work/notebooks/capstone.ipynb`

3. **Add your Hugging Face token to Colab Secrets:**
   - In Colab, click the key icon (🔑) in the left sidebar
   - Add a new secret named `HF_TOKEN` with your Hugging Face token as the value
   - Enable "Notebook access" for that secret

4. **Run all cells:**
   - `Runtime → Run all`
   - The notebook connects to the dataset, builds the modeling dataset, trains the models, and prints/generates all results and charts

That's it — no local Python setup, no downloads, no credentials beyond the one free Hugging Face token.

---

## Usage examples

Once the notebook is running, you can:

- **Reproduce the exact numbers in the paper** — every metric in the Results section comes directly from running this notebook top to bottom
- **Change the eligibility filters** — e.g., adjust the minimum impressions/clicks thresholds in Section 2 to see how the modeling universe changes
- **Try a different decision threshold** — Section 3.9 shows how precision/recall trade off across thresholds (0.05 to 0.50); pick the one that matches your own review capacity
- **Re-run feature importance** — Section 4.3 re-computes permutation importance any time you retrain the model, so you can check which signals matter most for your own data window

---

## Architecture sketch

```
FlyRank Hugging Face Dataset (79M+ row warehouse)
        │
        ▼
Feb 2026 partition ──► Aggregate GSC metrics per page
        │                (impressions, clicks, CTR, position)
        ▼
Join with content metadata (word count, backlinks, search volume, etc.)
        │
        ▼
Apply eligibility filters (published, not deleted, min traffic, created before window)
        │
        ▼
     Feb "universe" (29,700 pages)
        │
        ▼
Join with March 2026 outcomes ──► Label: went_dark = 1 if zero March clicks
        │
        ▼
   Final modeling dataset (29,353 pages, 3.95% positive rate)
        │
        ├──► Dummy Baseline (majority class)
        ├──► Logistic Regression (median impute + scale + balanced weights)
        └──► HistGradientBoosting (native missing-value handling)
        │
        ▼
Evaluate: Accuracy / Precision / Recall / F1 / PR-AUC / ROC-AUC
        │
        ▼
Permutation importance ──► Ranked recommendations
```

---

## Eval results (v2 — same split, model vs. baseline)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Dummy Baseline | 96.05% | 0.00% | 0.00% | 0.00% | — | — |
| Logistic Regression | 64.62% | 9.05% | 87.93% | 16.42% | 0.1323 | 0.8270 |
| HistGradientBoosting (t=0.50) | 96.05% | 50.00% | 0.43% | 0.85% | 0.2189 | 0.8700 |
| HistGradientBoosting (t=0.15) | — | 23.99% | 38.36% | 29.52% | 0.2189 | 0.8700 |

**Top predictive signals** (permutation importance, scored on PR-AUC):
1. `clicks_feb` (0.1141)
2. `impressions_feb` (0.0923)
3. `char_count` (0.0519)

Full results, charts, and interpretation are in the [deployed paper](https://prithvirajr203-pixel.github.io/Paper-Asisgnemnt-Flyrank-/).

---

## Limitations

- **Limited time window** — trained on one month predicting the next; may not generalize across seasons.
- **Severe class imbalance** — only 3.95% of pages went dark, so false positives remain substantial at any usable recall level.
- **Association, not causation** — feature importance shows what predicts went-dark risk, not what causes it.
- **Threshold sensitivity** — HistGradientBoosting's recall swings from 0.43% to 80.17% depending purely on the chosen decision threshold.
- **Missing features** — technical SEO issues, indexing status, and algorithm changes aren't in this dataset and could be real confounders.
- **Missing input data** — backlinks, word count, and char count had missing values, handled via imputation.
- **Single train/test split** — no cross-validation or temporal rolling validation performed.
- **Not yet operationally validated** — this model has never been used in a live SEO monitoring workflow.

Full limitations discussion in the [paper](https://prithvirajr203-pixel.github.io/Paper-Asisgnemnt-Flyrank-/).

---

## Built with AI

I built this project with Claude (Anthropic). Claude helped me: debug and fix broken notebook cells (the ranked-recommendations and artifact-generation sections), draft the structure and prose of the research paper and this README from my own model results, and troubleshoot GitHub Pages deployment issues. I ran the notebook myself, verified every number reported here against the actual notebook output, and made the modeling decisions (feature selection, leakage exclusions, threshold choices) myself.

---

## Links

- **Live paper:** https://prithvirajr203-pixel.github.io/Paper-Asisgnemnt-Flyrank-/
- **Repository:** https://github.com/prithvirajr203-pixel/Notebook-FlyRank
- **Notebook (Colab):** https://colab.research.google.com/github/prithvirajr203-pixel/Notebook-FlyRank/blob/main/work/notebooks/capstone.ipynb
- **Data source:** Built on the [FlyRank ML Internship dataset](https://flyrank.ai)
