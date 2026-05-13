"""
==============================================================
  STEP 2: Evaluate Base vs Fine-Tuned Model (Automated Metrics)
==============================================================
Computes ROUGE-L, BLEU, BERTScore, Cosine Similarity, and
Textbook Keyword Hit Rate for both models against ground truth.

Prerequisites:
    pip install rouge-score nltk bert-score sentence-transformers matplotlib

Usage:
    python evaluation/eval_base_vs_finetuned.py

Input:
    evaluation/results/model_responses.json  (from collect_responses.py)

Output:
    evaluation/results/metrics_summary.json
    evaluation/results/base_vs_finetuned_chart.png
    evaluation/results/per_question_details.json
"""

import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── Paths ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESPONSES_PATH = os.path.join(RESULTS_DIR, "model_responses.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_summary.json")
DETAILS_PATH = os.path.join(RESULTS_DIR, "per_question_details.json")
CHART_PATH = os.path.join(RESULTS_DIR, "base_vs_finetuned_chart.png")

os.makedirs(RESULTS_DIR, exist_ok=True)


def install_dependencies():
    """Install required packages if not present."""
    import subprocess
    import sys
    packages = [
        "rouge-score", "nltk", "bert-score",
        "sentence-transformers", "matplotlib"
    ]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


# ── Install deps ──────────────────────────────────────────
install_dependencies()

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ── Load Responses ─────────────────────────────────────────
print("📂 Loading model responses...")
with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
    responses = json.load(f)

print(f"   Loaded {len(responses)} question-answer pairs")


# ══════════════════════════════════════════════════════════
#  METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_bleu(reference, hypothesis):
    """Compute BLEU score with smoothing."""
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    if len(hyp_tokens) == 0:
        return 0.0

    smoothie = SmoothingFunction().method1
    # Use up to 4-gram, but handle short sequences
    weights = (0.25, 0.25, 0.25, 0.25)
    if len(ref_tokens) < 4 or len(hyp_tokens) < 4:
        weights = (0.5, 0.5)  # bigram BLEU for short texts

    return sentence_bleu([ref_tokens], hyp_tokens,
                         weights=weights,
                         smoothing_function=smoothie)


def compute_keyword_hit_rate(reference, hypothesis):
    """
    Compute what % of important keywords from the reference
    appear in the hypothesis. This directly measures textbook
    terminology usage.
    """
    # Extract meaningful words (skip common stop words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "it", "its", "this", "that", "these", "those", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "as", "into",
        "and", "or", "but", "not", "no", "if", "then", "than",
        "so", "up", "out", "about", "which", "who", "whom", "what",
        "when", "where", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "only",
        "very", "also", "just", "because", "between", "through",
    }

    ref_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', reference.lower()))
    hyp_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', hypothesis.lower()))

    # Filter out stop words → keep only "keywords"
    ref_keywords = ref_words - stop_words
    hyp_keywords = hyp_words - stop_words

    if len(ref_keywords) == 0:
        return 1.0

    hits = ref_keywords.intersection(hyp_keywords)
    return len(hits) / len(ref_keywords)


# ══════════════════════════════════════════════════════════
#  COMPUTE ALL METRICS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Computing Metrics...")
print("=" * 60)

# Prepare texts for BERTScore (batch computation)
references = [r["textbook_answer"] for r in responses]
finetuned_hyps = [r["finetuned_response"] for r in responses]
base_hyps = [r["base_response"] for r in responses]

# ── BERTScore (batch) ──────────────────────────────────────
print("\n📊 Computing BERTScore...")
ft_P, ft_R, ft_F1 = bert_score_fn(finetuned_hyps, references, lang="en", verbose=False)
base_P, base_R, base_F1 = bert_score_fn(base_hyps, references, lang="en", verbose=False)

# ── Cosine Similarity (batch) ─────────────────────────────
print("📊 Computing Cosine Similarity...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
ref_embeddings = embed_model.encode(references)
ft_embeddings = embed_model.encode(finetuned_hyps)
base_embeddings = embed_model.encode(base_hyps)

# ── Per-question metrics ──────────────────────────────────
print("📊 Computing ROUGE-L, BLEU, Keyword Hit Rate...")

per_question = []
ft_metrics = {"rouge_l": [], "bleu": [], "bert_f1": [], "cosine_sim": [], "keyword_hit": []}
base_metrics = {"rouge_l": [], "bleu": [], "bert_f1": [], "cosine_sim": [], "keyword_hit": []}

for i, r in enumerate(responses):
    ref = r["textbook_answer"]
    ft_resp = r["finetuned_response"]
    base_resp = r["base_response"]

    # ROUGE-L
    ft_rouge = compute_rouge_l(ref, ft_resp)
    base_rouge = compute_rouge_l(ref, base_resp)

    # BLEU
    ft_bleu = compute_bleu(ref, ft_resp)
    base_bleu = compute_bleu(ref, base_resp)

    # BERTScore (already computed)
    ft_bert = ft_F1[i].item()
    base_bert = base_F1[i].item()

    # Cosine Similarity
    ft_cos = cosine_similarity(
        ref_embeddings[i].reshape(1, -1),
        ft_embeddings[i].reshape(1, -1)
    )[0][0]
    base_cos = cosine_similarity(
        ref_embeddings[i].reshape(1, -1),
        base_embeddings[i].reshape(1, -1)
    )[0][0]

    # Keyword Hit Rate
    ft_kw = compute_keyword_hit_rate(ref, ft_resp)
    base_kw = compute_keyword_hit_rate(ref, base_resp)

    # Store
    ft_metrics["rouge_l"].append(ft_rouge)
    ft_metrics["bleu"].append(ft_bleu)
    ft_metrics["bert_f1"].append(ft_bert)
    ft_metrics["cosine_sim"].append(ft_cos)
    ft_metrics["keyword_hit"].append(ft_kw)

    base_metrics["rouge_l"].append(base_rouge)
    base_metrics["bleu"].append(base_bleu)
    base_metrics["bert_f1"].append(base_bert)
    base_metrics["cosine_sim"].append(base_cos)
    base_metrics["keyword_hit"].append(base_kw)

    per_question.append({
        "id": r["id"],
        "subject": r["subject"],
        "question": r["question"],
        "textbook_answer": ref,
        "finetuned_response": ft_resp,
        "base_response": base_resp,
        "finetuned_scores": {
            "rouge_l": round(ft_rouge, 4),
            "bleu": round(ft_bleu, 4),
            "bert_f1": round(ft_bert, 4),
            "cosine_sim": round(float(ft_cos), 4),
            "keyword_hit": round(ft_kw, 4)
        },
        "base_scores": {
            "rouge_l": round(base_rouge, 4),
            "bleu": round(base_bleu, 4),
            "bert_f1": round(base_bert, 4),
            "cosine_sim": round(float(base_cos), 4),
            "keyword_hit": round(base_kw, 4)
        }
    })

    winner = "✅ FT" if (ft_rouge + ft_bert + ft_cos + ft_kw) > (base_rouge + base_bert + base_cos + base_kw) else "⬜ Base"
    print(f"  [{r['subject'][:4]}] Q{i+1}: {winner} | "
          f"FT(R={ft_rouge:.2f} B={ft_bleu:.2f} BS={ft_bert:.2f} CS={ft_cos:.2f} KW={ft_kw:.2f}) | "
          f"Base(R={base_rouge:.2f} B={base_bleu:.2f} BS={base_bert:.2f} CS={base_cos:.2f} KW={base_kw:.2f})")


# ══════════════════════════════════════════════════════════
#  AGGREGATE SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  AGGREGATE RESULTS")
print("=" * 60)

summary = {
    "total_questions": len(responses),
    "finetuned_model": {
        "rouge_l_mean": round(np.mean(ft_metrics["rouge_l"]), 4),
        "bleu_mean": round(np.mean(ft_metrics["bleu"]), 4),
        "bert_f1_mean": round(np.mean(ft_metrics["bert_f1"]), 4),
        "cosine_sim_mean": round(float(np.mean(ft_metrics["cosine_sim"])), 4),
        "keyword_hit_mean": round(np.mean(ft_metrics["keyword_hit"]), 4),
    },
    "base_model": {
        "rouge_l_mean": round(np.mean(base_metrics["rouge_l"]), 4),
        "bleu_mean": round(np.mean(base_metrics["bleu"]), 4),
        "bert_f1_mean": round(np.mean(base_metrics["bert_f1"]), 4),
        "cosine_sim_mean": round(float(np.mean(base_metrics["cosine_sim"])), 4),
        "keyword_hit_mean": round(np.mean(base_metrics["keyword_hit"]), 4),
    },
    "improvement": {},
    "finetuned_wins": 0,
    "base_wins": 0,
    "ties": 0
}

# Calculate improvement percentages
for metric in ["rouge_l_mean", "bleu_mean", "bert_f1_mean", "cosine_sim_mean", "keyword_hit_mean"]:
    ft_val = summary["finetuned_model"][metric]
    base_val = summary["base_model"][metric]
    if base_val > 0:
        improvement = ((ft_val - base_val) / base_val) * 100
    else:
        improvement = 0
    summary["improvement"][metric.replace("_mean", "")] = round(improvement, 2)

# Count wins
for pq in per_question:
    ft_total = sum(pq["finetuned_scores"].values())
    base_total = sum(pq["base_scores"].values())
    if ft_total > base_total:
        summary["finetuned_wins"] += 1
    elif base_total > ft_total:
        summary["base_wins"] += 1
    else:
        summary["ties"] += 1

# Per-subject breakdown
subjects = list(set(r["subject"] for r in responses))
summary["per_subject"] = {}
for subj in subjects:
    subj_indices = [i for i, r in enumerate(responses) if r["subject"] == subj]
    summary["per_subject"][subj] = {
        "finetuned": {
            "rouge_l": round(np.mean([ft_metrics["rouge_l"][i] for i in subj_indices]), 4),
            "bleu": round(np.mean([ft_metrics["bleu"][i] for i in subj_indices]), 4),
            "bert_f1": round(np.mean([ft_metrics["bert_f1"][i] for i in subj_indices]), 4),
            "cosine_sim": round(float(np.mean([ft_metrics["cosine_sim"][i] for i in subj_indices])), 4),
            "keyword_hit": round(np.mean([ft_metrics["keyword_hit"][i] for i in subj_indices]), 4),
        },
        "base": {
            "rouge_l": round(np.mean([base_metrics["rouge_l"][i] for i in subj_indices]), 4),
            "bleu": round(np.mean([base_metrics["bleu"][i] for i in subj_indices]), 4),
            "bert_f1": round(np.mean([base_metrics["bert_f1"][i] for i in subj_indices]), 4),
            "cosine_sim": round(float(np.mean([base_metrics["cosine_sim"][i] for i in subj_indices])), 4),
            "keyword_hit": round(np.mean([base_metrics["keyword_hit"][i] for i in subj_indices]), 4),
        }
    }

# Print summary
print(f"\n{'Metric':<25} {'Fine-Tuned':>12} {'Base':>12} {'Improvement':>14}")
print("-" * 65)
for metric_key in ["rouge_l", "bleu", "bert_f1", "cosine_sim", "keyword_hit"]:
    ft_val = summary["finetuned_model"][f"{metric_key}_mean"]
    base_val = summary["base_model"][f"{metric_key}_mean"]
    imp = summary["improvement"][metric_key]
    label = metric_key.upper().replace("_", " ")
    print(f"  {label:<23} {ft_val:>12.4f} {base_val:>12.4f} {imp:>+12.1f}%")

print(f"\n  Fine-tuned wins: {summary['finetuned_wins']}/{summary['total_questions']}")
print(f"  Base wins:       {summary['base_wins']}/{summary['total_questions']}")
print(f"  Ties:            {summary['ties']}/{summary['total_questions']}")


# ══════════════════════════════════════════════════════════
#  GENERATE CHART
# ══════════════════════════════════════════════════════════
print("\n📊 Generating comparison chart...")

metrics_labels = ["ROUGE-L", "BLEU", "BERTScore\nF1", "Cosine\nSimilarity", "Keyword\nHit Rate"]
ft_values = [
    summary["finetuned_model"]["rouge_l_mean"],
    summary["finetuned_model"]["bleu_mean"],
    summary["finetuned_model"]["bert_f1_mean"],
    summary["finetuned_model"]["cosine_sim_mean"],
    summary["finetuned_model"]["keyword_hit_mean"],
]
base_values = [
    summary["base_model"]["rouge_l_mean"],
    summary["base_model"]["bleu_mean"],
    summary["base_model"]["bert_f1_mean"],
    summary["base_model"]["cosine_sim_mean"],
    summary["base_model"]["keyword_hit_mean"],
]

x = np.arange(len(metrics_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

bars1 = ax.bar(x - width/2, base_values, width, label="Base Mistral-7B",
               color="#e94560", edgecolor="white", linewidth=0.5, alpha=0.9)
bars2 = ax.bar(x + width/2, ft_values, width, label="Fine-Tuned (LoRA)",
               color="#0f3460", edgecolor="white", linewidth=0.5, alpha=0.9)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color="#e94560", fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color="#53d8fb", fontweight="bold")

ax.set_xlabel("Evaluation Metric", fontsize=12, color="white", labelpad=10)
ax.set_ylabel("Score", fontsize=12, color="white", labelpad=10)
ax.set_title("Base Mistral-7B vs Fine-Tuned Model\n(NEB Textbook Answer Alignment)",
             fontsize=14, color="white", fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=10, color="white")
ax.tick_params(axis='y', colors='white')
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", fontsize=11, facecolor="#16213e", edgecolor="white",
          labelcolor="white")
ax.grid(axis="y", alpha=0.2, color="white")

# Remove border
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"   Chart saved: {CHART_PATH}")

# ── Per-Subject Chart ──────────────────────────────────────
subject_chart_path = os.path.join(RESULTS_DIR, "per_subject_chart.png")

fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 5))
fig.patch.set_facecolor("#1a1a2e")

if len(subjects) == 1:
    axes = [axes]

subject_colors = {"Physics": "#00b4d8", "Chemistry": "#90be6d", "Computer Science": "#f9c74f"}

for idx, subj in enumerate(sorted(subjects)):
    ax = axes[idx]
    ax.set_facecolor("#16213e")

    subj_data = summary["per_subject"][subj]
    metrics_short = ["ROUGE-L", "BLEU", "BERTScore", "CosSim", "KeyHit"]
    ft_vals = [subj_data["finetuned"][k] for k in ["rouge_l", "bleu", "bert_f1", "cosine_sim", "keyword_hit"]]
    base_vals = [subj_data["base"][k] for k in ["rouge_l", "bleu", "bert_f1", "cosine_sim", "keyword_hit"]]

    x2 = np.arange(len(metrics_short))
    ax.bar(x2 - 0.15, base_vals, 0.3, color="#e94560", alpha=0.8, label="Base")
    ax.bar(x2 + 0.15, ft_vals, 0.3, color=subject_colors.get(subj, "#53d8fb"), alpha=0.8, label="Fine-Tuned")

    ax.set_title(subj, fontsize=12, color="white", fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics_short, fontsize=8, color="white", rotation=30)
    ax.tick_params(axis='y', colors='white')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor="#16213e", edgecolor="white", labelcolor="white")
    ax.grid(axis="y", alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("Per-Subject Evaluation: Base vs Fine-Tuned", fontsize=14, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(subject_chart_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"   Per-subject chart saved: {subject_chart_path}")


# ══════════════════════════════════════════════════════════
#  SAVE FILES
# ══════════════════════════════════════════════════════════
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(DETAILS_PATH, "w", encoding="utf-8") as f:
    json.dump(per_question, f, indent=2, ensure_ascii=False)

print(f"\n✅ Metrics summary saved: {METRICS_PATH}")
print(f"✅ Per-question details saved: {DETAILS_PATH}")
print(f"\n{'=' * 60}")
print(f"  EVALUATION COMPLETE!")
print(f"{'=' * 60}")
