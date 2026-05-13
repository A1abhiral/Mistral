"""
==============================================================
  STEP 3: Comparative Analysis with External AI Systems
==============================================================
After you manually collect responses from ChatGPT, Gemini, and
Khanmigo, add them to:
    evaluation/results/external_responses.json

Then run this script to generate comparison tables and charts.

Usage:
    1. First run: python evaluation/comparative_analysis.py --template
       (generates a template JSON for you to fill in)
    2. Fill in external responses manually
    3. Then run: python evaluation/comparative_analysis.py
       (generates analysis + charts)
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
QUESTIONS_PATH = os.path.join(SCRIPT_DIR, "comparison_questions.json")
MODEL_RESPONSES_PATH = os.path.join(RESULTS_DIR, "model_responses.json")
EXTERNAL_PATH = os.path.join(RESULTS_DIR, "external_responses.json")
COMPARATIVE_OUTPUT = os.path.join(RESULTS_DIR, "comparative_analysis.json")
COMPARATIVE_CHART = os.path.join(RESULTS_DIR, "comparative_chart.png")
SIDE_BY_SIDE_PATH = os.path.join(RESULTS_DIR, "side_by_side_comparison.md")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  TEMPLATE GENERATION MODE
# ══════════════════════════════════════════════════════════
if "--template" in sys.argv:
    print("📝 Generating template for external responses...")

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    template = []
    for q in questions:
        template.append({
            "id": q["id"],
            "subject": q["subject"],
            "question": q["question"],
            "textbook_answer": q["textbook_answer"],
            "chatgpt_response": "<<PASTE CHATGPT RESPONSE HERE>>",
            "gemini_response": "<<PASTE GEMINI RESPONSE HERE>>",
            "khanmigo_response": "<<PASTE KHANMIGO RESPONSE HERE or REMOVE THIS FIELD>>"
        })

    with open(EXTERNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Template saved to: {EXTERNAL_PATH}")
    print(f"\n📌 INSTRUCTIONS:")
    print(f"   1. Open {EXTERNAL_PATH}")
    print(f"   2. For each question, ask ChatGPT, Gemini, and Khanmigo")
    print(f"   3. Paste their responses in the respective fields")
    print(f"   4. IMPORTANT: When asking, provide the EXACT question text")
    print(f"   5. When done, run: python evaluation/comparative_analysis.py")
    print(f"\n💡 TIP: You can skip Khanmigo if you don't have access.")
    print(f"   Just remove the 'khanmigo_response' field from each entry.")
    sys.exit(0)


# ══════════════════════════════════════════════════════════
#  ANALYSIS MODE
# ══════════════════════════════════════════════════════════
print("📊 Running Comparative Analysis...")

# Check files exist
if not os.path.exists(MODEL_RESPONSES_PATH):
    print("❌ model_responses.json not found. Run collect_responses.py first.")
    sys.exit(1)

if not os.path.exists(EXTERNAL_PATH):
    print("❌ external_responses.json not found. Run with --template first.")
    sys.exit(1)

# Load data
with open(MODEL_RESPONSES_PATH, "r", encoding="utf-8") as f:
    model_responses = json.load(f)

with open(EXTERNAL_PATH, "r", encoding="utf-8") as f:
    external_responses = json.load(f)

# Detect which external systems have responses
external_systems = []
sample = external_responses[0]
for key in ["chatgpt_response", "gemini_response", "khanmigo_response"]:
    if key in sample and "<<PASTE" not in sample[key]:
        system_name = key.replace("_response", "").upper()
        if system_name == "CHATGPT":
            system_name = "ChatGPT"
        elif system_name == "GEMINI":
            system_name = "Gemini"
        elif system_name == "KHANMIGO":
            system_name = "Khanmigo"
        external_systems.append((key, system_name))

print(f"   External systems detected: {[s[1] for s in external_systems]}")

# Install deps
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Compute Metrics for All Systems ──────────────────────
def compute_system_metrics(references, hypotheses):
    """Compute all metrics for a system's responses."""
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]

    # BERTScore
    _, _, bert_f1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    bert_scores = bert_f1.tolist()

    # Cosine Similarity
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb = embed_model.encode(references)
    hyp_emb = embed_model.encode(hypotheses)
    cos_scores = [
        float(cosine_similarity(ref_emb[i].reshape(1, -1), hyp_emb[i].reshape(1, -1))[0][0])
        for i in range(len(references))
    ]

    return {
        "rouge_l": round(np.mean(rouge_scores), 4),
        "bert_f1": round(np.mean(bert_scores), 4),
        "cosine_sim": round(np.mean(cos_scores), 4),
        "avg_combined": round(np.mean([np.mean(rouge_scores), np.mean(bert_scores), np.mean(cos_scores)]), 4)
    }


# Build reference list
references = [m["textbook_answer"] for m in model_responses]
finetuned_hyps = [m["finetuned_response"] for m in model_responses]
base_hyps = [m["base_response"] for m in model_responses]

all_systems = {}

print("\n  Computing metrics for Fine-Tuned model...")
all_systems["Fine-Tuned\n(Ours)"] = compute_system_metrics(references, finetuned_hyps)

print("  Computing metrics for Base Mistral-7B...")
all_systems["Base\nMistral-7B"] = compute_system_metrics(references, base_hyps)

for key, name in external_systems:
    ext_hyps = [e[key] for e in external_responses]
    print(f"  Computing metrics for {name}...")
    all_systems[name] = compute_system_metrics(references, ext_hyps)


# ── Print Comparison Table ────────────────────────────────
print("\n" + "=" * 80)
print("  COMPARATIVE RESULTS (All Systems vs NEB Textbook Answers)")
print("=" * 80)

print(f"\n{'System':<20} {'ROUGE-L':>10} {'BERTScore':>10} {'CosSim':>10} {'Average':>10}")
print("-" * 62)
for name, metrics in all_systems.items():
    display_name = name.replace("\n", " ")
    print(f"  {display_name:<18} {metrics['rouge_l']:>10.4f} {metrics['bert_f1']:>10.4f} "
          f"{metrics['cosine_sim']:>10.4f} {metrics['avg_combined']:>10.4f}")


# ── Generate Comparative Chart ────────────────────────────
print("\n📊 Generating comparative chart...")

system_names = list(all_systems.keys())
metrics_to_plot = ["rouge_l", "bert_f1", "cosine_sim"]
metric_labels = ["ROUGE-L", "BERTScore F1", "Cosine Similarity"]

# Color palette
colors = {
    "Fine-Tuned\n(Ours)": "#00b4d8",
    "Base\nMistral-7B": "#e94560",
    "ChatGPT": "#74b9ff",
    "Gemini": "#a29bfe",
    "Khanmigo": "#fdcb6e"
}

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

x = np.arange(len(metric_labels))
n_systems = len(system_names)
total_width = 0.7
bar_width = total_width / n_systems

for i, sys_name in enumerate(system_names):
    values = [all_systems[sys_name][m] for m in metrics_to_plot]
    offset = (i - n_systems / 2 + 0.5) * bar_width
    color = colors.get(sys_name, f"#{hash(sys_name) % 0xFFFFFF:06x}")
    bars = ax.bar(x + offset, values, bar_width, label=sys_name.replace("\n", " "),
                  color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color="white", fontweight="bold")

ax.set_xlabel("Metric", fontsize=12, color="white", labelpad=10)
ax.set_ylabel("Score (higher = more textbook-aligned)", fontsize=11, color="white", labelpad=10)
ax.set_title("Comparative Evaluation: All Systems vs NEB Textbook Answers",
             fontsize=14, color="white", fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11, color="white")
ax.tick_params(axis='y', colors='white')
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", fontsize=9, facecolor="#16213e", edgecolor="white",
          labelcolor="white", ncol=2)
ax.grid(axis="y", alpha=0.15, color="white")
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(COMPARATIVE_CHART, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Chart saved: {COMPARATIVE_CHART}")


# ── Generate Side-by-Side Markdown ────────────────────────
print("\n📝 Generating side-by-side comparison document...")

md_lines = ["# Side-by-Side Response Comparison\n"]
md_lines.append("This document compares responses from all systems for each test question.\n")
md_lines.append(f"**Systems compared:** {', '.join(s.replace(chr(10), ' ') for s in system_names)}\n")
md_lines.append("---\n")

for i, q in enumerate(model_responses):
    md_lines.append(f"## Q{i+1}: {q['question']}")
    md_lines.append(f"**Subject:** {q['subject']}\n")
    md_lines.append(f"### 📖 Textbook Answer")
    md_lines.append(f"> {q['textbook_answer']}\n")
    md_lines.append(f"### 🧪 Fine-Tuned Model (Ours)")
    md_lines.append(f"{q['finetuned_response']}\n")
    md_lines.append(f"### 🔹 Base Mistral-7B")
    md_lines.append(f"{q['base_response']}\n")

    ext = external_responses[i]
    for key, name in external_systems:
        md_lines.append(f"### 🌐 {name}")
        md_lines.append(f"{ext[key]}\n")

    md_lines.append("---\n")

with open(SIDE_BY_SIDE_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"   Side-by-side document saved: {SIDE_BY_SIDE_PATH}")


# ── Save Results ──────────────────────────────────────────
with open(COMPARATIVE_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_systems, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✅ Comparative analysis saved: {COMPARATIVE_OUTPUT}")
print(f"\n{'=' * 60}")
print(f"  COMPARATIVE ANALYSIS COMPLETE!")
print(f"{'=' * 60}")
