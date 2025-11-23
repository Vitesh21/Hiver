import argparse
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def normalize_text(subject: str, body: str) -> str:
    text = f"{subject} \n {body}".lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_rule_patterns() -> Dict[str, List[Tuple[re.Pattern, str]]]:
    patterns = {
        "CUST_A": [
            (re.compile(r"shared mailbox|permission denied|access"), "access_issue"),
            (re.compile(r"rule (not working|stopped|no longer|not triggering)|auto-assign|refund"), "workflow_issue"),
            (re.compile(r"stuck in pending|pending.*resolved"), "status_bug"),
            (re.compile(r"thread(s)? not merging|different threads"), "threading_issue"),
            (re.compile(r"tag suggestions incorrect|irrelevant tags"), "tagging_accuracy"),
            (re.compile(r"drafts disappearing|draft replies disappear"), "ui_bug"),
            (re.compile(r"automation delay|taking \d+\s*minutes"), "automation_delay"),
            (re.compile(r"login issues|invalid session"), "auth_issue"),
            (re.compile(r"export not downloading|csv fails"), "export_issue"),
            (re.compile(r"notification spam|duplicate desktop notifications"), "notification_bug"),
            (re.compile(r"feature request"), "feature_request"),
        ],
        "CUST_B": [
            (re.compile(r"duplicate tasks|2 tasks|duplication"), "automation_bug"),
            (re.compile(r"tags missing|tags not saved|not appearing"), "tagging_issue"),
            (re.compile(r"billing mismatch|charged incorrectly|invoice"), "billing_error"),
            (re.compile(r"slow email load|\bslow\b|\bdelay\b"), "performance"),
            (re.compile(r"mobile app crash|crashes"), "mobile_bug"),
            (re.compile(r"sla not applying|sla rules"), "sla_issue"),
            (re.compile(r"incorrect user assignments|wrong agent"), "assignment_bug"),
            (re.compile(r"imap sync failure|sync halted"), "sync_issue"),
            (re.compile(r"feature request|unified analytics"), "feature_request"),
        ],
        "CUST_C": [
            (re.compile(r"csat (not visible|disappeared|not appearing)|csat"), "analytics_issue"),
            (re.compile(r"delay|takes \d+\s*[-–]\s*\d+ seconds|\bslow\b"), "performance"),
            (re.compile(r"configure sla|setting up sla|slas"), "setup_help"),
            (re.compile(r"mail merge (stuck|failing)"), "mail_merge_issue"),
            (re.compile(r"search not returning results"), "search_issue"),
            (re.compile(r"deleted emails reappearing|reappear"), "sync_bug"),
            (re.compile(r"tables broken|composer|editor"), "editor_bug"),
            (re.compile(r"attachments corrupted|corrupted"), "attachment_issue"),
            (re.compile(r"auto-close not working"), "automation_issue"),
            (re.compile(r"csat survey not sent"), "csat_issue"),
            (re.compile(r"ui freeze|freezes"), "ui_performance"),
            (re.compile(r"delay in notifications|notifications .* minutes late"), "notification_delay"),
            (re.compile(r"dark mode request|dark mode"), "feature_request"),
        ],
        "CUST_D": [
            (re.compile(r"mail merge"), "mail_merge_issue"),
            (re.compile(r"add new user|authorization required|authorization missing|add user"), "user_management"),
            (re.compile(r"feature request"), "feature_request"),
            (re.compile(r"archived emails missing|analytics"), "analytics_bug"),
            (re.compile(r"kanban view glitch|cards overlap"), "ui_bug"),
            (re.compile(r"forwarding fails|server timeout"), "forwarding_issue"),
            (re.compile(r"signature duplication|duplicate twice"), "signature_bug"),
            (re.compile(r"custom fields lost|disappear after switching tabs"), "ui_state_bug"),
            (re.compile(r"report export incorrect|slas look incorrect"), "analytics_accuracy"),
            (re.compile(r"smart suggestions irrelevant|wrong kb"), "suggestion_accuracy"),
            (re.compile(r"confetti animation stuck|plays repeatedly"), "ui_bug"),
        ],
    }
    return patterns


def build_anti_patterns() -> Dict[str, Dict[str, List[re.Pattern]]]:
    anti = {
        "CUST_B": {
            "performance": [re.compile(r"automation")],  # prefer automation_* if automation present
            "feature_request": [re.compile(r"error|fail|bug|issue")],  # if strong failure words, avoid feature_request
        },
        "CUST_A": {
            "workflow_issue": [re.compile(r"billing")],
            "ui_bug": [re.compile(r"login|auth")],  # avoid misclassifying auth issues
        },
        "CUST_C": {
            "performance": [re.compile(r"csat|survey")],  # analytics-related should avoid performance
        },
        "CUST_D": {
            "ui_bug": [re.compile(r"analytics|report")],  # avoid analytics issues tagged as ui_bug
        },
    }
    return anti


class CustomerModel:
    def __init__(self, tags: List[str]):
        self.tags = sorted(list(set(tags)))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.clf = LogisticRegression(max_iter=1000, multi_class="auto")

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def predict_proba(self, text: str) -> Dict[str, float]:
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        return {cls: float(p) for cls, p in zip(self.clf.classes_, probs)}

    def predict(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        return self.clf.predict(X)[0]


def apply_rules(customer_id: str, text: str, patterns: Dict[str, List[Tuple[re.Pattern, str]]]) -> str:
    for pat, tag in patterns.get(customer_id, []):
        if pat.search(text):
            return tag
    return ""


def apply_anti_patterns(customer_id: str, text: str, proba: Dict[str, float], anti: Dict[str, Dict[str, List[re.Pattern]]]) -> Dict[str, float]:
    modified = proba.copy()
    for tag, pats in anti.get(customer_id, {}).items():
        for pat in pats:
            if pat.search(text):
                # Downweight misleading tag probabilities
                modified[tag] = modified.get(tag, 0.0) * 0.5
    # Renormalize
    total = sum(modified.values())
    if total > 0:
        for k in modified:
            modified[k] /= total
    return modified


def train_per_customer(df: pd.DataFrame) -> Dict[str, CustomerModel]:
    models = {}
    for cust, group in df.groupby("customer_id"):
        texts = [normalize_text(s, b) for s, b in zip(group["subject"], group["body"])]
        labels = list(group["tag"])
        model = CustomerModel(tags=labels)
        model.fit(texts, labels)
        models[cust] = model
    return models


def evaluate(df: pd.DataFrame, cust_models: Dict[str, CustomerModel], patterns, anti) -> None:
    print("\n=== Evaluation (per customer train/test split) ===")
    for cust, group in df.groupby("customer_id"):
        if len(group) < 2:
            print(f"{cust}: not enough samples to evaluate")
            continue
        # Use stratify only when all classes have at least 2 samples
        tag_counts = group["tag"].value_counts()
        can_stratify = len(set(group["tag"])) > 1 and tag_counts.min() >= 2
        if can_stratify:
            train, test = train_test_split(group, test_size=0.33, random_state=42, stratify=group["tag"]) 
        else:
            train, test = train_test_split(group, test_size=0.33, random_state=42)
        # Train model
        model = CustomerModel(tags=list(train["tag"]))
        train_texts = [normalize_text(s, b) for s, b in zip(train["subject"], train["body"])]
        model.fit(train_texts, list(train["tag"]))

        # Test
        y_true, y_pred = [], []
        for _, row in test.iterrows():
            text = normalize_text(row["subject"], row["body"])
            # Rule first
            tag = apply_rules(cust, text, patterns)
            if tag:
                y_pred.append(tag)
            else:
                proba = model.predict_proba(text)
                proba = apply_anti_patterns(cust, text, proba, anti)
                pred = max(proba.items(), key=lambda kv: kv[1])[0]
                y_pred.append(pred)
            y_true.append(row["tag"])

        print(f"\nCustomer: {cust}")
        print(classification_report(y_true, y_pred, zero_division=0))


def predict_single(df: pd.DataFrame, cust_models: Dict[str, CustomerModel], customer_id: str, subject: str, body: str, patterns, anti) -> str:
    text = normalize_text(subject, body)
    tag = apply_rules(customer_id, text, patterns)
    if tag:
        return tag
    model = cust_models[customer_id]
    proba = model.predict_proba(text)
    proba = apply_anti_patterns(customer_id, text, proba, anti)
    return max(proba.items(), key=lambda kv: kv[1])[0]


def main():
    parser = argparse.ArgumentParser(description="Part A: Email Tagging with Customer Isolation")
    parser.add_argument("--data", type=str, default="data/small_dataset.csv", help="Path to CSV dataset")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    patterns = build_rule_patterns()
    anti = build_anti_patterns()

    # Train models per customer on full dataset
    cust_models = train_per_customer(df)

    # Evaluate with simple per-customer split
    evaluate(df, cust_models, patterns, anti)

    # Demo predictions showing isolation
    print("\n=== Demo Predictions (isolated per customer) ===")
    demos = [
        ("CUST_A", "Permission denied on shared mailbox", "Cannot access shared mailbox today."),
        ("CUST_B", "Automation duplicating tasks", "After edit, every email creates two tasks."),
        ("CUST_C", "CSAT missing", "CSAT scores not appearing since morning."),
        ("CUST_D", "User add fails", "Authorization required when adding user."),
    ]
    for cust, subj, body in demos:
        pred = predict_single(df, cust_models, cust, subj, body, patterns, anti)
        print(f"Customer={cust} | Subject='{subj}' => Predicted tag: {pred}")


if __name__ == "__main__":
    main()