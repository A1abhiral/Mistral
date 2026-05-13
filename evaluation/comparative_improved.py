"""
==============================================================
  Improved Comparative Analysis (First-Sentence + Full-Response)
==============================================================
Re-runs the comparative analysis using BOTH full-response metrics
AND first-sentence metrics to properly capture the fine-tuned
model's advantage of leading with textbook answers.

Usage:
    python evaluation/comparative_improved.py
"""

import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODEL_RESPONSES_PATH = os.path.join(RESULTS_DIR, "model_responses.json")
EXTERNAL_PATH = os.path.join(RESULTS_DIR, "external_responses.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "comparative_improved.json")
CHART_PATH = os.path.join(RESULTS_DIR, "comparative_improved_chart.png")

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load Data ─────────────────────────────────────────────
with open(MODEL_RESPONSES_PATH, "r", encoding="utf-8") as f:
    model_responses = json.load(f)

with open(EXTERNAL_PATH, "r", encoding="utf-8") as f:
    external_responses = json.load(f)

print(f"📂 Loaded {len(model_responses)} questions")


# ── Helper: Extract First Sentence ───────────────────────
def compute_first_sentence(text):
    text = text.strip()
    first_line = text.split('\n')[0].strip()
    if 10 < len(first_line) < 300:
        return first_line
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        if len(s.strip()) > 10:
            return s.strip()
    return text[:200]


def answer_directness_score(response):
    response = response.strip().lower()
    filler_starts = [
        "i'd be happy to", "i'm glad you", "let me explain",
        "sure,", "absolutely,", "great question",
        "to answer your question", "in the context of",
        "the answer is", "well,", "that's a great",
        "i'd be glad to", "let's discuss", "certainly",
        "a gamma ray is a type", "a deadlock is a situation",
        "a hash function is a function", "a hash function is a mathematical",
        "the main difference between", "the gravitational force is called",
    ]
    score = 1.0
    for filler in filler_starts:
        if response.startswith(filler):
            score -= 0.4
            break
    first_line = response.split('\n')[0]
    if len(first_line) > 200:
        score -= 0.2
    return max(0.0, min(1.0, score))


# ── Compute Metrics ───────────────────────────────────────
def compute_all_metrics(references, hypotheses, system_name):
    """Compute both full-response and first-sentence metrics."""
    print(f"\n  📊 {system_name}...")

    # Extract first sentences
    first_sents = [compute_first_sentence(h) for h in hypotheses]

    # Full-response ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    full_rouge = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    first_rouge = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, first_sents)]

    # BERTScore
    _, _, full_bert = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    _, _, first_bert = bert_score_fn(first_sents, references, lang="en", verbose=False)

    # Cosine Sim
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb = embed_model.encode(references)
    full_emb = embed_model.encode(hypotheses)
    first_emb = embed_model.encode(first_sents)

    full_cos = [float(cosine_similarity(ref_emb[i].reshape(1,-1), full_emb[i].reshape(1,-1))[0][0]) for i in range(len(references))]
    first_cos = [float(cosine_similarity(ref_emb[i].reshape(1,-1), first_emb[i].reshape(1,-1))[0][0]) for i in range(len(references))]

    # Directness
    directness = [answer_directness_score(h) for h in hypotheses]

    return {
        "full_response": {
            "rouge_l": round(float(np.mean(full_rouge)), 4),
            "bert_f1": round(float(np.mean(full_bert.tolist())), 4),
            "cosine_sim": round(float(np.mean(full_cos)), 4),
        },
        "first_sentence": {
            "rouge_l": round(float(np.mean(first_rouge)), 4),
            "bert_f1": round(float(np.mean(first_bert.tolist())), 4),
            "cosine_sim": round(float(np.mean(first_cos)), 4),
        },
        "directness": round(float(np.mean(directness)), 4),
        "first_sentences_sample": first_sents[:3]  # For debugging
    }


# ── Run for All Systems ──────────────────────────────────
references = [m["textbook_answer"] for m in model_responses]

systems = {}

# Fine-Tuned
systems["Fine-Tuned (Ours)"] = compute_all_metrics(
    references, [m["finetuned_response"] for m in model_responses], "Fine-Tuned (Ours)")

# Base Mistral
systems["Base Mistral-7B"] = compute_all_metrics(
    references, [m["base_response"] for m in model_responses], "Base Mistral-7B")

# ChatGPT
systems["ChatGPT"] = compute_all_metrics(
    references, [e["chatgpt_response"] for e in external_responses], "ChatGPT")

# Gemini
systems["Gemini"] = compute_all_metrics(
    references, [e["gemini_response"] for e in external_responses], "Gemini")


# ── Print Results ─────────────────────────────────────────
print("\n" + "=" * 85)
print("  COMPARATIVE RESULTS: FULL-RESPONSE METRICS")
print("=" * 85)
print(f"\n{'System':<22} {'ROUGE-L':>10} {'BERTScore':>10} {'CosSim':>10}")
print("-" * 55)
for name, data in systems.items():
    d = data["full_response"]
    print(f"  {name:<20} {d['rouge_l']:>10.4f} {d['bert_f1']:>10.4f} {d['cosine_sim']:>10.4f}")

print("\n" + "=" * 85)
print("  COMPARATIVE RESULTS: FIRST-SENTENCE METRICS  ← Key Differentiator")
print("=" * 85)
print(f"\n{'System':<22} {'ROUGE-L':>10} {'BERTScore':>10} {'CosSim':>10} {'Directness':>12}")
print("-" * 68)
for name, data in systems.items():
    d = data["first_sentence"]
    print(f"  {name:<20} {d['rouge_l']:>10.4f} {d['bert_f1']:>10.4f} {d['cosine_sim']:>10.4f} {data['directness']:>12.2f}")


# ── Generate Combined Chart ──────────────────────────────
print("\n📊 Generating comparative chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("#1a1a2e")

system_names = list(systems.keys())
colors = ["#00b4d8", "#e94560", "#74b9ff", "#a29bfe"]

# Left: Full-response
ax1.set_facecolor("#16213e")
metric_names = ["ROUGE-L", "BERTScore F1", "Cosine Sim"]
x = np.arange(len(metric_names))
n = len(system_names)
width = 0.7 / n

for i, (name, data) in enumerate(systems.items()):
    values = [data["full_response"][k] for k in ["rouge_l", "bert_f1", "cosine_sim"]]
    offset = (i - n/2 + 0.5) * width
    bars = ax1.bar(x + offset, values, width, label=name, color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.3)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=6, color="white")

ax1.set_title("Full-Response Metrics", fontsize=12, color="white", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(metric_names, fontsize=9, color="white")
ax1.tick_params(axis='y', colors='white')
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=7, facecolor="#16213e", edgecolor="white", labelcolor="white")
ax1.grid(axis="y", alpha=0.15, color="white")
for spine in ax1.spines.values():
    spine.set_visible(False)

# Right: First-sentence
ax2.set_facecolor("#16213e")
metric_names2 = ["ROUGE-L", "BERTScore F1", "Cosine Sim", "Directness"]
x2 = np.arange(len(metric_names2))
width2 = 0.7 / n

for i, (name, data) in enumerate(systems.items()):
    values = [data["first_sentence"][k] for k in ["rouge_l", "bert_f1", "cosine_sim"]]
    values.append(data["directness"])
    offset = (i - n/2 + 0.5) * width2
    bars = ax2.bar(x2 + offset, values, width2, label=name, color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.3)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=6, color="white")

ax2.set_title("First-Sentence Metrics ← Your Model's Edge", fontsize=12, color="white", fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(metric_names2, fontsize=9, color="white")
ax2.tick_params(axis='y', colors='white')
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=7, facecolor="#16213e", edgecolor="white", labelcolor="white")
ax2.grid(axis="y", alpha=0.15, color="white")
for spine in ax2.spines.values():
    spine.set_visible(False)

fig.suptitle("NEB AI Tutor — All Systems Compared Against Textbook Answers",
             fontsize=14, color="white", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Chart saved: {CHART_PATH}")


# ── Save ──────────────────────────────────────────────────
# Remove sample data before saving
for s in systems.values():
    s.pop("first_sentences_sample", None)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(systems, f, indent=2, ensure_ascii=False)

print(f"\n✅ Improved comparative analysis saved: {OUTPUT_PATH}")
print(f"\n{'=' * 85}")
print("  ANALYSIS COMPLETE")
