"""
==============================================================
  Generate Printable Human Evaluation Survey
==============================================================
Takes model responses and creates a blind survey where
System A, B, C are randomly shuffled per question.

Usage:
    python evaluation/generate_survey.py

Input:
    evaluation/results/model_responses.json
    evaluation/results/external_responses.json (optional)

Output:
    evaluation/results/human_eval_survey.md     (for evaluators)
    evaluation/results/survey_answer_key.json   (for you — maps A/B/C to systems)
"""

import json
import os
import random

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODEL_RESPONSES_PATH = os.path.join(RESULTS_DIR, "model_responses.json")
EXTERNAL_PATH = os.path.join(RESULTS_DIR, "external_responses.json")
SURVEY_PATH = os.path.join(RESULTS_DIR, "human_eval_survey.md")
ANSWER_KEY_PATH = os.path.join(RESULTS_DIR, "survey_answer_key.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────
with open(MODEL_RESPONSES_PATH, "r", encoding="utf-8") as f:
    model_responses = json.load(f)

# Check for external responses
has_external = os.path.exists(EXTERNAL_PATH)
external_responses = []
if has_external:
    with open(EXTERNAL_PATH, "r", encoding="utf-8") as f:
        external_responses = json.load(f)
    # Check if they're actually filled in
    sample = external_responses[0]
    if "<<PASTE" in sample.get("chatgpt_response", ""):
        has_external = False
        print("⚠️  External responses not filled in yet. Using 3 systems: Base, Fine-Tuned, ChatGPT placeholder.")

print(f"📋 Loaded {len(model_responses)} questions")
print(f"   External responses: {'Yes' if has_external else 'No (will use Base + Fine-Tuned only)'}")

# ── Build Survey ──────────────────────────────────────────
md = []
answer_key = []

md.append("# NEB AI Tutor — Human Evaluation Survey")
md.append("")
md.append("## Instructions")
md.append("")
md.append("You will evaluate answers from **3 AI systems** (labeled **System A**, **System B**, **System C**).")
md.append("You do **NOT** know which system produced which answer.")
md.append("")
md.append("**Rate each answer on a scale of 1-5** for each criterion:")
md.append("")
md.append("| Score | Meaning |")
md.append("|:-----:|---------|")
md.append("| **1** | Very Poor — Wrong or misleading |")
md.append("| **2** | Poor — Partially correct, significant errors |")
md.append("| **3** | Average — Roughly correct but generic |")
md.append("| **4** | Good — Correct, mostly NEB-aligned |")
md.append("| **5** | Excellent — Matches NEB textbook precisely |")
md.append("")
md.append("## Evaluator Information")
md.append("")
md.append("| Field | Response |")
md.append("|-------|----------|")
md.append("| **Name** | |")
md.append("| **Role** | ☐ Teacher  ☐ Student  ☐ Subject Expert |")
md.append("| **Subject Expertise** | ☐ Physics  ☐ Chemistry  ☐ Computer Science |")
md.append("")
md.append("---")
md.append("")

for i, q in enumerate(model_responses):
    # Build systems for this question
    systems = {
        "Fine-Tuned (Ours)": q["finetuned_response"],
        "Base Mistral-7B": q["base_response"],
    }

    if has_external:
        ext = external_responses[i]
        if "chatgpt_response" in ext and "<<PASTE" not in ext.get("chatgpt_response", ""):
            systems["ChatGPT"] = ext["chatgpt_response"]
        if "gemini_response" in ext and "<<PASTE" not in ext.get("gemini_response", ""):
            systems["Gemini"] = ext["gemini_response"]

    # If we have more than 3, pick: Fine-Tuned, Base, and one external
    system_names = list(systems.keys())

    # Shuffle to randomize A/B/C assignment
    random.shuffle(system_names)

    # Take up to 3 systems
    selected = system_names[:3]

    labels = ["A", "B", "C"]
    question_key = {
        "question_num": i + 1,
        "question": q["question"],
        "subject": q["subject"],
        "mapping": {}
    }

    md.append(f"## Question {i + 1}")
    md.append(f"**Subject:** {q['subject']}")
    md.append(f"")
    md.append(f"**Question:** {q['question']}")
    md.append(f"")
    md.append(f"**Reference (Textbook Answer):**")
    md.append(f"> {q['textbook_answer']}")
    md.append(f"")

    for j, sys_name in enumerate(selected):
        label = labels[j]
        response = systems[sys_name]
        question_key["mapping"][f"System {label}"] = sys_name

        md.append(f"### System {label}")
        md.append(f"")
        md.append(f"> {response}")
        md.append(f"")
        md.append(f"| Criterion | Score (1-5) |")
        md.append(f"|-----------|:-----------:|")
        md.append(f"| Correctness | |")
        md.append(f"| NEB Curriculum Alignment | |")
        md.append(f"| Appropriate Depth | |")
        md.append(f"| NEB Terminology | |")
        md.append(f"| Exam Relevance | |")
        md.append(f"")

    answer_key.append(question_key)
    md.append("---")
    md.append("")

# Overall preferences
md.append("## Overall Preferences")
md.append("")
md.append("1. **Which system provided the most NEB-aligned answers?**")
md.append("   - ☐ System A  ☐ System B  ☐ System C")
md.append("")
md.append("2. **Which system would you recommend for NEB exam preparation?**")
md.append("   - ☐ System A  ☐ System B  ☐ System C")
md.append("")
md.append("3. **Additional Comments:** _______________")
md.append("")
md.append("---")
md.append("*Thank you for your evaluation!*")

# ── Save ──────────────────────────────────────────────────
with open(SURVEY_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

with open(ANSWER_KEY_PATH, "w", encoding="utf-8") as f:
    json.dump(answer_key, f, indent=2, ensure_ascii=False)

print(f"\n✅ Survey saved: {SURVEY_PATH}")
print(f"✅ Answer key saved: {ANSWER_KEY_PATH}")
print(f"\n📌 Share {SURVEY_PATH} with evaluators.")
print(f"🔒 Keep {ANSWER_KEY_PATH} private — it maps A/B/C to actual systems.")
print(f"\n💡 TIP: Convert the survey to a Google Form for easier collection!")
