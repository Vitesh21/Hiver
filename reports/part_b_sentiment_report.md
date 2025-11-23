# Part B – Sentiment Prompt Evaluation Report

## Overview
We evaluated a sentiment prompt on 10 support emails, targeting consistent labels, confidence calibration, and hidden reasoning for debugging.

## Prompt v1 Failures
- Ambiguous cases (generic “not working”, “unable”) sometimes returned neutral due to insufficient guidance.
- Overconfidence on weak signals; lacked explicit calibration policy.
- Formatting risks without strict JSON schema enforcement.

## Prompt v2 Improvements
- Deterministic policy and conflict resolution (negative overrides neutral when both present).
- Confidence tied to indicator strength, capped at 0.95.
- Strict JSON output requirements reduce formatting drift.

## Results Summary
- Heuristic stand-in produced more consistent negatives for explicit failures (e.g., “charged incorrectly”, “permission denied”, “disappeared”).
- Feature requests and setup help generally scored neutral with moderate confidence.

## Systematic Evaluation Guidance
- Create a labeled gold set; annotate sentiment and evidence phrases.
- Track error buckets: mixed signals, weak cues, domain-specific phrasing.
- Use agreement metrics and calibration curves; inspect rationales.
- Iterate policies: add indicator dictionaries, conflict rules, and validation checks.

## Next Steps
- Add domain lexicons and phrase-level patterns.
- Introduce JSON schema validation pre-consumption.
- Optional: few-shot examples in the prompt to stabilize behavior.