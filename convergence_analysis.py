# -*- coding: utf-8 -*-

import json
import textwrap
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os

#  Paths
BASE = (
    "."
)
EVAL_DIR      = f"{BASE}/Evaluation Dataset"
TAXONOMY_JSON = f"{BASE}/error_taxonomy_annotated.json"
OUT_DIR       = f"{BASE}/convergence_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "Bengali":  f"{EVAL_DIR}/bengali_evaluations.csv",
    "Italian":  f"{EVAL_DIR}/italian_evaluations.csv",
    "Hindi":    f"{EVAL_DIR}/hindi_evaluations.csv",
    "Urdu":     f"{EVAL_DIR}/urdu_evaluations.csv",
    "Punjabi":  f"{EVAL_DIR}/punjabi_evaluations.csv",
    "Sicilian": f"{EVAL_DIR}/sicilian_evaluations.csv",
    "Sindhi":   f"{EVAL_DIR}/sindhi_evaluations.csv",
}

LANG_META = {
    "Italian":  {"tier": "high", "region": "Italy"},
    "Hindi":    {"tier": "high", "region": "India"},
    "Bengali":  {"tier": "mid",  "region": "India"},
    "Urdu":     {"tier": "mid",  "region": "Pakistan"},
    "Punjabi":  {"tier": "low",  "region": "India"},
    "Sicilian": {"tier": "low",  "region": "Italy"},
    "Sindhi":   {"tier": "low",  "region": "Pakistan"},
}
TIER_ORDER = ["high", "mid", "low"]

# Language display order: high → mid → low, alpha within tier
LANG_DISPLAY = sorted(
    LANG_META.keys(),
    key=lambda l: ({"high": 0, "mid": 1, "low": 2}[LANG_META[l]["tier"]], l)
)

#  Taxonomy exclusions (mirrors Error_taxonomy2 notebook)
TAX_EXCLUDE = {
    "No Math Issue", "No grammar issue", "No cultural mistake",
    "Drop response - invalid", "Unclear or unspecified issue",
    "Did not understand instructions",
}

# Taxonomy label → super-category (Grammar / Culture / Math)
SUPER = {
    "Fluency, awkwardness, and redundancy":      "Grammar",
    "Word choice and lexical mistranslation":     "Grammar",
    "Unspecified grammar issue":                  "Grammar",
    "Word order and sentence structure":          "Grammar",
    "Register, formality, and tone":              "Grammar",
    "Language complexity and age level":          "Grammar",
    "Other grammar or language issue":            "Grammar",
    "Spelling, orthography, and diacritics":      "Grammar",
    "Tense":                                      "Grammar",
    "Pronoun and unclear reference":              "Grammar",
    "Agreement ":                                 "Grammar",
    "Style":                                      "Grammar",
    "Prepositions and postposition":              "Grammar",
    "Article":                                    "Grammar",
    "Numerals":                                   "Grammar",
    "Punctuation and non-math symbols":           "Grammar",
    "Unspecified cultural mismatch":              "Culture",
    "Name form":                                  "Culture",
    "Unrealistic or implausible scenerio":        "Culture",
    "Currency and denominations":                 "Culture",
    "Overgeneralization or stereotype":           "Culture",
    "Age-appropriateness":                        "Culture",
    "Language choice for local math convention":  "Culture",
    "Ettiquette and respectful norms":            "Culture",
    "Other cultural mismatch":                    "Culture",
    "Economic context: prices and affordability": "Culture",
    "Over-localization":                          "Culture",
    "Material culture objectss":                  "Culture",
    "Food and drink items":                       "Culture",
    "Humor":                                      "Culture",
    "Traditional activities and games":           "Culture",
    "Attributed item from the wrong culture":     "Culture",
    "Location or Geography":                      "Culture",
    "Culturally insentitive":                     "Culture",
    "Religion, ritual, and myth":                 "Culture",
    "Regional-linguistic mismatch":               "Culture",
    "Movies, shows or characters":                "Culture",
    "Emotion":                                    "Culture",
    "Completely madeup item that doesn't exist in culture": "Culture",
    "Math problem confusing":                     "Math",
    "Unspecified Math Issue":                     "Math",
    "Conversion and rate errors":                 "Math",
    "Context completely changed":                 "Math",
    "Units and quantities":                       "Math",
    "Units and denominations errors":             "Math",
    "Numeric value errors":                       "Math",
    "Internal contradiction in scenario":         "Math",
    "Operation or relationship errors":           "Math",
    "Other Math Issue":                           "Math",
    "Target quantity ambiguous or misleading":    "Math",
    "Symbolic and notation errors":               "Math",
}

# Checklist items
CHECKLIST_LABELS = [
    "Names of people, places, or objects did not reflect the target culture",
    "Settings or locations did not feel realistic for the region",
    "Social values or norms (e.g., family roles, gender roles, authority) were not culturally appropriate",
    "Age appropriateness - text was not aligned with expectations for school-aged children",
    "Everyday practices (e.g., school system, sports, holidays, foods, religious references) were inaccurate or unfamiliar",
    "Communication style (e.g., politeness, directness, tone) did not fit the cultural norms",
    "Idioms, metaphors, or expressions were confusing or inappropriate",
    "Humor was not culturally appropriate or understandable",
    "Units, currency, or assumptions (e.g., measurement systems, pricing) did not match the culture",
    "Format or problem type does not resemble what is typically used in local math education",
    "Content that may be perceived as culturally insensitive or offensive",
    "The translation relied on generalized or overly simplistic cultural representations",
    "The translation did not reflect the diversity or complexity of the culture",
    "Phrases or vocabulary felt awkward or not typical of how people actually speak in the local context",
    "None of the above, all aspects of the text felt culturally appropriate",
    "Other (please specify):",
]
CHK_SUBSTANTIVE = set(CHECKLIST_LABELS) - {
    "None of the above, all aspects of the text felt culturally appropriate",
    "Other (please specify):",
}

#  Plot style
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
})

# Measure colours
M_COLORS = {
    "likert":    "#2166AC",   # blue
    "checklist": "#7b4fa6",   # purple
    "taxonomy":  "#c0392b",   # red
}
M_LABELS = {
    "likert":    "Likert composite",
    "checklist": "Checklist flag rate",
    "taxonomy":  "Taxonomy error rate",
}

#  Survey CSVs
dfs = []
for lang, fpath in FILES.items():
    df = pd.read_csv(fpath)
    df["language_label"] = lang
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)
print(f"Survey rows loaded: {len(full_df):,}")

#  Likert wide format
likert = full_df[full_df["Question_No_Qualtrics"].str.endswith("_1", na=False)].copy()
likert["response_num"] = pd.to_numeric(likert["Response"], errors="coerce")

q = likert["Question_Qualtrics"].str.lower().fillna("")
likert["dimension"] = "unknown"
likert.loc[q.str.contains("grammar|natural flow"),            "dimension"] = "LQ"
likert.loc[q.str.contains("culturally appropriate"),          "dimension"] = "CA"
likert.loc[q.str.contains("mathematical reasoning|preserve"), "dimension"] = "RP"

wide_lik = (
    likert[likert["dimension"] != "unknown"]
    .pivot_table(
        index=["ResponseId", "problem_number", "model", "language_label"],
        columns="dimension",
        values="response_num",
        aggfunc="mean",
    )
    .reset_index()
)
wide_lik.columns.name = None
wide_lik["problem_number"] = pd.to_numeric(wide_lik["problem_number"], errors="coerce")
wide_lik["composite_lik"] = wide_lik[["LQ", "CA", "RP"]].mean(axis=1)
print(f"wide_lik rows: {len(wide_lik):,}  "
      f"problems: {wide_lik['problem_number'].nunique()}  "
      f"languages: {wide_lik['language_label'].nunique()}")

#  Checklist rows
checklist_df = full_df[
    full_df["Question_Qualtrics"].str.contains("Which aspects|did not align",
                                               case=False, na=False)
].copy()
checklist_df["problem_number"] = pd.to_numeric(
    checklist_df["problem_number"], errors="coerce"
)
print(f"Checklist rows: {len(checklist_df):,}")

#  Taxonomy JSON
with open(TAXONOMY_JSON) as f:
    raw = json.load(f)

tax_rows = []
for record in raw:
    d = record["data"]
    for ann in record["annotations"]:
        if ann["was_cancelled"]:
            continue
        for result in ann["result"]:
            if "choices" not in result["value"]:
                continue
            for choice in result["value"]["choices"]:
                if choice in TAX_EXCLUDE:
                    continue
                tax_rows.append({
                    "ResponseId":     d["ResponseId"],
                    "problem_number": int(d["problem_number"]),
                    "model":          d["model"],
                    "language":       d["language"],
                    "label":          choice,
                    "super_category": SUPER.get(choice, "Other"),
                })

tax_df = pd.DataFrame(tax_rows).drop_duplicates(
    subset=["ResponseId", "problem_number", "model", "label"]
).reset_index(drop=True)
print(f"Taxonomy rows (substantive, deduped): {len(tax_df):,}")

# Compute per-problem metrics (overall and by language)

#  Helper: parse checklist response into flag count
def count_flags(response_str):
    r = str(response_str)
    return sum(
        1 for label in sorted(CHECKLIST_LABELS, key=len, reverse=True)
        if label.rstrip(":") in r and label in CHK_SUBSTANTIVE
    )

checklist_df["n_flags"] = checklist_df["Response"].fillna("").apply(count_flags)

#  OVERALL metrics
# Likert
lik_overall = (
    wide_lik.groupby("problem_number")
    .agg(lik_score=("composite_lik", "mean"),
         lik_n=("composite_lik", "count"))
    .reset_index()
)

# Checklist
chk_overall = (
    checklist_df.groupby("problem_number")
    .agg(chk_rate=("n_flags", "mean"),
         chk_n=("n_flags", "count"))
    .reset_index()
)

# Taxonomy: labels coded per evaluation
# Numerator = unique (ResponseId, label) pairs per problem
tax_per_prob = (
    tax_df.drop_duplicates(["ResponseId", "problem_number", "label"])
    .groupby("problem_number")
    .size()
    .reset_index(name="tax_labels")
)
# Denominator from wide_lik (all evaluations)
eval_per_prob = (
    wide_lik.groupby("problem_number")
    .size()
    .reset_index(name="n_evals")
)
tax_overall = eval_per_prob.merge(tax_per_prob, on="problem_number", how="left")
tax_overall["tax_labels"] = tax_overall["tax_labels"].fillna(0)
tax_overall["tax_rate"] = tax_overall["tax_labels"] / tax_overall["n_evals"]

# Merge
overall = (
    lik_overall
    .merge(chk_overall,  on="problem_number", how="outer")
    .merge(tax_overall[["problem_number", "tax_rate", "n_evals"]],
           on="problem_number", how="outer")
)
print(f"Overall table: {len(overall)} problems")

#  BY-LANGUAGE metrics
lik_bylang = (
    wide_lik.groupby(["problem_number", "language_label"])
    .agg(lik_score=("composite_lik", "mean"),
         lik_n=("composite_lik", "count"))
    .reset_index()
)

chk_bylang = (
    checklist_df.groupby(["problem_number", "language_label"])
    .agg(chk_rate=("n_flags", "mean"),
         chk_n=("n_flags", "count"))
    .reset_index()
)

tax_bylang = (
    tax_df.drop_duplicates(["ResponseId", "problem_number", "language", "label"])
    .groupby(["problem_number", "language"])
    .size()
    .reset_index(name="tax_labels")
    .rename(columns={"language": "language_label"})
)
eval_bylang = (
    wide_lik.groupby(["problem_number", "language_label"])
    .size()
    .reset_index(name="n_evals")
)
tax_bylang_full = eval_bylang.merge(tax_bylang, on=["problem_number", "language_label"], how="left")
tax_bylang_full["tax_labels"] = tax_bylang_full["tax_labels"].fillna(0)
tax_bylang_full["tax_rate"] = tax_bylang_full["tax_labels"] / tax_bylang_full["n_evals"]

bylang = (
    lik_bylang
    .merge(chk_bylang,  on=["problem_number", "language_label"], how="outer")
    .merge(tax_bylang_full[["problem_number", "language_label", "tax_rate", "n_evals"]],
           on=["problem_number", "language_label"], how="outer")
)
print(f"By-language table: {len(bylang)} rows")
print(f"  problems: {bylang['problem_number'].nunique()}  "
      f"  languages: {bylang['language_label'].nunique()}")

#Normalise to badness percentile ranks + convergence score

def prank(series, ascending=True):
    #Percentile rank 0–1. ascending=True means high value → high rank.
    return series.rank(pct=True, na_option="keep", ascending=ascending)

#  Overall
overall["lik_bad"]   = 1 - prank(overall["lik_score"],  ascending=True)
overall["chk_bad"]   = prank(overall["chk_rate"],   ascending=True)
overall["tax_bad"]   = prank(overall["tax_rate"],   ascending=True)
overall["convergence"] = overall[["lik_bad", "chk_bad", "tax_bad"]].mean(axis=1)

THRESHOLD = 0.75   # "bottom quartile" on a given measure
overall["lik_flagged"]  = (overall["lik_bad"]  >= THRESHOLD).astype(int)
overall["chk_flagged"]  = (overall["chk_bad"]  >= THRESHOLD).astype(int)
overall["tax_flagged"]  = (overall["tax_bad"]  >= THRESHOLD).astype(int)
overall["n_measures_flagged"] = (
    overall["lik_flagged"] + overall["chk_flagged"] + overall["tax_flagged"]
)

overall_sorted = overall.sort_values("convergence", ascending=False).reset_index(drop=True)

#  By language (ranks computed within each language separately)
for col, asc in [("lik_score", True), ("chk_rate", True), ("tax_rate", True)]:
    bad_col = col.split("_")[0] + "_bad"
    if col == "lik_score":
        bylang[bad_col] = bylang.groupby("language_label")[col].transform(
            lambda s: 1 - prank(s, ascending=True)
        )
    else:
        bylang[bad_col] = bylang.groupby("language_label")[col].transform(
            lambda s: prank(s, ascending=True)
        )

bylang["convergence"]   = bylang[["lik_bad", "chk_bad", "tax_bad"]].mean(axis=1)
bylang["lik_flagged"]   = (bylang["lik_bad"]  >= THRESHOLD).astype(int)
bylang["chk_flagged"]   = (bylang["chk_bad"]  >= THRESHOLD).astype(int)
bylang["tax_flagged"]   = (bylang["tax_bad"]  >= THRESHOLD).astype(int)
bylang["n_measures_flagged"] = (
    bylang["lik_flagged"] + bylang["chk_flagged"] + bylang["tax_flagged"]
)

#  Print overall summary
print("\nProblems by how many measures flagged them (≥ bottom quartile):\n")
vc = overall["n_measures_flagged"].value_counts().sort_index(ascending=False)
for k, v in vc.items():
    stars = "★" * k
    label = {3: "all three  ← most concerning",
             2: "two of three",
             1: "one only",
             0: "none"}.get(k, "")
    print(f"  {k} measures {stars:<4}  {v:3d} problems   {label}")

print("\nTop 10 most convergently worst problems:")
print(f"{'Q#':>4}  {'Conv':>6}  {'lik_bad':>8}  {'chk_bad':>8}  "
      f"{'tax_bad':>8}  {'flagged':>8}")
print("-" * 55)
for _, r in overall_sorted.head(10).iterrows():
    print(f"  {int(r['problem_number']):>4}  {r['convergence']:.3f}  "
          f"{r['lik_bad']:.3f}     {r['chk_bad']:.3f}     "
          f"{r['tax_bad']:.3f}     {int(r['n_measures_flagged'])}/3")

#  Save CSVs
overall_sorted.to_csv(f"{OUT_DIR}/convergence_scores_overall.csv", index=False)
bylang.sort_values(["language_label", "convergence"], ascending=[True, False]).to_csv(
    f"{OUT_DIR}/convergence_scores_by_language.csv", index=False
)

TOP_N = 15

top15 = overall_sorted.head(TOP_N).reset_index(drop=True)
top15_plot = top15.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(3.03, 3.4))
fig.subplots_adjust(left=0.12, right=0.82, top=0.92, bottom=0.12)

measure_cols = [("lik_bad",  "likert"),
                ("chk_bad",  "checklist"),
                ("tax_bad",  "taxonomy")]
OFFSETS = {"likert": -0.18, "checklist": 0.0, "taxonomy": 0.18}

for i, (_, row) in enumerate(top15_plot.iterrows()):
    for col, key in measure_cols:
        val = row[col]
        if not np.isnan(val):
            ax.scatter(
                val, i + OFFSETS[key],
                color=M_COLORS[key], s=30,
                edgecolors="white", linewidths=0.3, zorder=3, alpha=0.9
            )
    vals = [row[c] for c, _ in measure_cols if not np.isnan(row[c])]
    if len(vals) >= 2:
        ax.plot([min(vals), max(vals)], [i, i],
                color="#cccccc", linewidth=0.8, zorder=2)

ax.axvline(THRESHOLD, color="#aaaaaa", linewidth=0.7, linestyle="--", alpha=0.8)
ax.set_yticks(range(TOP_N))
ax.set_yticklabels(
    [f"Q{int(r['problem_number'])}" for _, r in top15_plot.iterrows()],
    fontsize=8
)
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.6, TOP_N - 0.4)
ax.set_xlabel("Badness percentile rank  (1 = worst)", fontsize=8)
ax.set_title(f"Top {TOP_N} worst problems", fontsize=8, fontweight="bold",
             pad=5, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.2, linewidth=0.4)

handles = [
    mpatches.Patch(color=M_COLORS[k], alpha=0.9, label=M_LABELS[k])
    for k in ["likert", "checklist", "taxonomy"]
]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
          frameon=False, fontsize=7)

fig.savefig(f"{OUT_DIR}/figure_worst_problems_profile.pdf",
            bbox_inches="tight", format="pdf")
fig.savefig(f"{OUT_DIR}/figure_worst_problems_profile.png",
            bbox_inches="tight", dpi=300)
plt.close()

# Per-language convergence: top-5 worst per language
print(f"\n{'=':=<72}")
print("WORST-CONVERGENT PROBLEMS BY LANGUAGE  (top 5)")
print(f"{'=':=<72}")
print(f"{'Language':<12} {'Q#':>4}  {'Conv':>6}  {'lik_bad':>8}  "
      f"{'chk_bad':>8}  {'tax_bad':>8}  {'flagged':>8}")
print("-" * 65)

for lang in LANG_DISPLAY:
    sub = (
        bylang[bylang["language_label"] == lang]
        .sort_values("convergence", ascending=False)
        .head(5)
    )
    for _, r in sub.iterrows():
        print(
            f"{lang:<12} {int(r['problem_number']):>4}  "
            f"{r['convergence']:.3f}  "
            f"{r['lik_bad']:.3f}     {r['chk_bad']:.3f}     "
            f"{r['tax_bad']:.3f}     {int(r['n_measures_flagged'])}/3"
        )
    print()

#  Problems triple-flagged across the most languages
triple_by_lang = (
    bylang[bylang["n_measures_flagged"] == 3]
    .groupby("problem_number")
    .size()
    .reset_index(name="n_langs_triple_flagged")
)
top_cross = (
    triple_by_lang.merge(
        overall[["problem_number", "convergence", "lik_score", "chk_rate", "tax_rate"]],
        on="problem_number"
    )
    .sort_values("n_langs_triple_flagged", ascending=False)
    .head(10)
)
print(f"\n{'=':=<72}")
print("PROBLEMS TRIPLE-FLAGGED ACROSS THE MOST LANGUAGES")
print(f"{'=':=<72}")
print(f"{'Q#':>4}  {'# langs':>8}  {'Conv':>7}  {'lik_score':>10}  "
      f"{'chk_rate':>9}  {'tax_rate':>9}")
print("-" * 58)
for _, r in top_cross.iterrows():
    print(
        f"{int(r['problem_number']):>4}  "
        f"{int(r['n_langs_triple_flagged']):>8}  "
        f"{r['convergence']:>7.3f}  "
        f"{r['lik_score']:>10.3f}  "
        f"{r['chk_rate']:>9.3f}  "
        f"{r['tax_rate']:>9.3f}"
    )

