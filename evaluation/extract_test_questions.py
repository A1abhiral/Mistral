"""
Extract a stratified sample of test questions from val.jsonl
for the evaluation framework.
"""
import json
import random
import os

random.seed(42)

# Read validation data
val_path = os.path.join(os.path.dirname(__file__), "..", "data", "val.jsonl")
with open(val_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse and group by subject
by_subject = {}
for line in lines:
    entry = json.loads(line.strip())
    system_msg = entry["messages"][0]["content"]
    user_msg = entry["messages"][1]["content"]
    assistant_msg = entry["messages"][2]["content"]

    # Extract subject from user message
    subject = "Unknown"
    if "Physics" in system_msg:
        subject = "Physics"
    elif "Chemistry" in system_msg:
        subject = "Chemistry"
    elif "Computer" in system_msg:
        subject = "Computer Science"

    by_subject.setdefault(subject, []).append({
        "system": system_msg,
        "question": user_msg,
        "textbook_answer": assistant_msg,
        "subject": subject
    })

print("=== Subject Distribution in Validation Set ===")
for subj, items in by_subject.items():
    print(f"  {subj}: {len(items)} questions")

# Sample 6 per subject (18 total) — diverse mix
test_questions = []
for subj, items in by_subject.items():
    # Pick 6 diverse questions
    sample = random.sample(items, min(6, len(items)))
    for i, item in enumerate(sample):
        test_questions.append({
            "id": f"{subj.lower().replace(' ', '_')}_{i+1}",
            "subject": subj,
            "question": item["question"],
            "textbook_answer": item["textbook_answer"],
            "system_prompt": item["system"]
        })

# Save
output_path = os.path.join(os.path.dirname(__file__), "comparison_questions.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(test_questions, f, indent=2, ensure_ascii=False)

print(f"\n✅ Extracted {len(test_questions)} test questions → comparison_questions.json")
for q in test_questions:
    print(f"  [{q['subject']}] {q['question'][:80]}...")
