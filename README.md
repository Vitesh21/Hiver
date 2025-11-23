# Hiver – AI Intern Evaluation Assignment

This repository implements all parts (A, B, C) using reproducible Python scripts and sample data/KB files. It runs end-to-end locally without external APIs, with optional hooks to use open-source embeddings.

## How to Run

- Prerequisites: Python 3.9+
- Install dependencies:

```
pip install -r requirements.txt
```

- Part A (Email Tagging):

```
python scripts/part_a_email_tagging.py --data data/small_dataset.csv
```

- Part B (Sentiment Prompt Evaluation):

```
python scripts/part_b_sentiment_prompt_eval.py --data data/small_dataset.csv --n 10
```

- Part C (Mini-RAG over KB):

```
python scripts/part_c_mini_rag.py --kb_dir kb
```

## Part A — Email Tagging Mini-System

### Approach

- Build per-customer text classifiers to ensure strict isolation of tags.
- For each `customer_id`, train a TF‑IDF + Logistic Regression multi-class model on that customer’s emails (`subject + body`).
- At prediction time, only tags belonging to the same customer are considered (no leakage).
- Augment with pattern-based rules (keywords/regex) and anti-pattern guardrails to improve accuracy.

### Model / Prompt

- Baseline: TF‑IDF vectorizer + Logistic Regression per customer.
- Rule layer:
  - Patterns: Regex-based triggers like “shared mailbox” → `access_issue` for CUST_A.
  - Anti-patterns: Words that commonly mislead the model reduce probability for a tag (e.g., “automation” reduces `performance` likelihood in CUST_B).
- Optional LLM prompt classifier stub is provided (disabled by default) to demonstrate prompt-based design without requiring keys.

### Customer Isolation

- Training isolation: Separate models per `customer_id`. Each model only sees that customer’s data and labels.
- Prediction isolation: Given an email with `customer_id`, route to the matching customer-specific model and restrict candidate tags to that customer’s set.

### Error Analysis

- Observed patterns:
  - Clear keyword cues (e.g., “permissions”, “duplicate tasks”, “CSAT”) strongly improve precision.
  - Short subjects benefit from including body text.
- Anti-patterns / guardrails:
  - “automation” present with “delay” → prefer `automation_delay` over `performance`.
  - “billing” appearing in non-billing customers → downweight misleading tag if ever in candidate set.
- Failure modes:
  - Very small per-customer datasets (3 examples) can overfit; rules mitigate but cannot fully solve.
  - Ambiguous wording without explicit domain words (e.g., “not working” generic) relies on context.

### 3 Productionization Improvements

- Add weak supervision: Use label functions and distant supervision to expand training via rules and heuristics.
- Introduce calibration and confidence thresholds with abstain/triage to human reviewers.
- Use domain-tuned embeddings or fine-tuned models per customer with transfer learning and regularization.

## Part B — Sentiment Analysis Prompt Evaluation

### Prompt v1

```
You are a sentiment analysis assistant for customer support emails.
Return a JSON object with fields:
- sentiment: one of [positive, negative, neutral]
- confidence: float in [0,1] calibrated to your certainty
- reasoning: a brief explanation for internal debugging (do not mention in end-user output)

Guidelines:
- Base sentiment only on the user’s language in subject/body.
- Treat requests and bug reports as negative only if frustration/impact is explicit.
- Neutral for informational or feature requests without emotion.
- Calibrate confidence: weak evidence <0.6, strong explicit complaints >0.8.
Return strictly valid JSON.
```

### Prompt v2 (Improved)

```
Role: Deterministic sentiment rater for support emails.
Output: Strict JSON with keys {"sentiment","confidence","reasoning"}.
Policy:
- Sentiment labels: {positive, negative, neutral}.
- Negative indicators: explicit frustration, errors, outages, incorrect charges, delays, inability to perform tasks.
- Neutral indicators: requests, setup help, generic questions, feature suggestions without sentiment.
- Positive indicators: appreciation, compliments, successful outcomes.
Consistency:
- If both negative and neutral indicators appear, choose negative and explain dominance.
Calibration:
- Confidence proportional to indicator strength and count; cap at 0.95.
Validation:
- Ensure valid JSON, lowercase labels, confidence in [0,1].
Return only JSON.
```

### What Failed in v1

- Edge cases with mixed signals produced inconsistent neutral vs negative.
- Confidence not consistently calibrated; overconfident on weak evidence.

### What Improved in v2

- Clear conflict resolution rule (negative dominates over neutral if present).
- Calibration policy capped and tied to evidence strength.
- Deterministic structure reduces formatting drift.

### Systematic Prompt Evaluation

- Create small gold set and annotate ground truth manually.
- Measure agreement metrics and error buckets (mixed signals, weak context, formatting errors).
- Iterate: add decision rules, calibration policy, and validation checks.

## Part C — Mini-RAG for Knowledge Base Answering

### Approach

- Index KB articles using embeddings:
  - Prefer `sentence-transformers` (`all-MiniLM-L6-v2`) if available.
  - Fall back to TF‑IDF cosine similarity when embeddings aren’t installed.
- Retrieve top-k articles per query, generate a concise answer using the retrieved context, and compute a confidence score from similarity margins.

### Queries and Outputs

- “How do I configure automations in Hiver?”
- “Why is CSAT not appearing?”

### Confidence Score

- Calculated from top similarity and margin to the next result; higher margin yields higher confidence.

### Retrieval Improvements (5)

- Use multi-vector retrieval (title + body + FAQ snippets).
- Add metadata filters (feature flags, plan tier, product area).
- Chunking with overlap and passage-level embeddings.
- Re-ranking with cross-encoders for better precision.
- Answer grounding with citations and coverage checks over multiple passages.

### Failure Case and Debugging

- Failure: Query uses synonyms not present in KB (e.g., “NPS not visible” but KB says “CSAT”).
- Debugging: Add synonym normalization, expand vocabulary with domain glossary, and include alias metadata on indexing.

## Notes

- The large dataset has been expanded to include rows 1–39 based on your latest input. Line 40 in the message was incomplete, so it is not included to keep the CSV valid. Small dataset is complete.
- Scripts print concise outputs demonstrating each part end-to-end.