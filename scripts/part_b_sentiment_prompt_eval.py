import argparse
from typing import Dict, List
import re
import json
import pandas as pd


PROMPT_V1 = r"""
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
"""

PROMPT_V2 = r"""
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
"""


NEGATIVE_WORDS = set("unable error fail failing stuck delay incorrect outage crash denied missing duplicated issue bug not working charged incorrectly disappeared".split())
POSITIVE_WORDS = set("thanks appreciate great resolved success".split())


def heuristic_sentiment(text: str) -> Dict[str, str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)

    # Determine label
    if neg_count > 0 and neg_count >= pos_count:
        label = "negative"
    elif pos_count > 0 and pos_count > neg_count:
        label = "positive"
    else:
        label = "neutral"

    # Confidence calibration
    strength = max(neg_count, pos_count)
    if strength == 0:
        confidence = 0.5
        rationale = "No strong sentiment indicators detected; defaulting to neutral."
    else:
        confidence = min(0.95, 0.6 + 0.1 * strength)
        rationale = f"Indicators: negative={neg_count}, positive={pos_count}."

    return {
        "sentiment": label,
        "confidence": round(confidence, 2),
        "reasoning": rationale,
    }


def run_prompt_eval(df: pd.DataFrame, n: int):
    rows = df.head(n)
    print("\n=== Prompt v1 (specification) ===\n")
    print(PROMPT_V1)
    print("\n=== Prompt v2 (improved specification) ===\n")
    print(PROMPT_V2)

    print("\n=== Heuristic Results (stand-in for LLM) ===")
    results_v1: List[Dict] = []
    results_v2: List[Dict] = []

    for _, r in rows.iterrows():
        text = f"{r['subject']}\n{r['body']}"
        out_v1 = heuristic_sentiment(text)
        out_v2 = heuristic_sentiment(text)
        # v2 tweak: if both neutral indicators and clear negative words like 'charged incorrectly', push to negative
        if re.search(r"charged incorrectly|permission denied|not working|fail|stuck|disappeared|crash|outage", text.lower()):
            if out_v2["sentiment"] != "negative":
                out_v2["sentiment"] = "negative"
                out_v2["confidence"] = max(out_v2["confidence"], 0.8)
                out_v2["reasoning"] += " Forced negative due to strong failure indicators."

        results_v1.append({"email_id": r["email_id"], **out_v1})
        results_v2.append({"email_id": r["email_id"], **out_v2})

    print("\nPrompt v1 results:")
    print(json.dumps(results_v1, indent=2))
    print("\nPrompt v2 results:")
    print(json.dumps(results_v2, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Part B: Sentiment Prompt Evaluation")
    parser.add_argument("--data", type=str, default="data/small_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--n", type=int, default=10, help="Number of emails to evaluate")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    run_prompt_eval(df, args.n)
    print("\nSee reports/part_b_sentiment_report.md for analysis and guidance.")


if __name__ == "__main__":
    main()