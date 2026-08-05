# -*- coding: utf-8 -*-


import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
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

BASE = ""

TAXONOMY_JSON = f"{BASE}/error_taxonomy_annotated.json"
EVAL_DIR      = f"{BASE}/Evaluation Dataset"
OUT_DIR       = f"{BASE}/error_taxonomy"


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

with open(TAXONOMY_JSON) as f:
    raw = json.load(f)

rows = []
for record in raw:
    d = record["data"]
    for ann in record["annotations"]:
        if ann["was_cancelled"]:
            continue
        for result in ann["result"]:
            if "choices" not in result["value"]:
                continue
            for choice in result["value"]["choices"]:
                rows.append({
                    "ResponseId":     d["ResponseId"],
                    "problem_number": int(d["problem_number"]),
                    "model":          d["model"],
                    "language":       d["language"],
                    "region":         d["region"],
                    "q_short_label":  d["q_short_label"],
                    "ambiguity":      d.get("Ambiguity", 0),
                    "no_elements":    d.get("No Elements", None),
                    "label_category": result["from_name"],
                    "label":          choice,
                })

tax_df = pd.DataFrame(rows)
print(f"Taxonomy rows (before any dedup/filter): {len(tax_df)}")

dfs = []
for lang, fpath in FILES.items():
    df = pd.read_csv(fpath)
    df["language_label"] = lang
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)

likert = full_df[full_df["Question_No_Qualtrics"].str.endswith("_1", na=False)].copy()
likert["response_num"] = pd.to_numeric(likert["Response"], errors="coerce")

q = likert["Question_Qualtrics"].str.lower().fillna("")
likert["dimension"] = "unknown"
likert.loc[q.str.contains("grammar|natural flow"),            "dimension"] = "LQ"
likert.loc[q.str.contains("culturally appropriate"),          "dimension"] = "CA"
likert.loc[q.str.contains("mathematical reasoning|preserve"), "dimension"] = "RP"

likert["resource_tier"] = likert["language_label"].map(
    {k: v["tier"] for k, v in LANG_META.items()}
)

wide_lik = likert[likert["dimension"] != "unknown"].pivot_table(
    index=["ResponseId", "problem_number", "model", "language_label", "resource_tier"],
    columns="dimension",
    values="response_num",
).reset_index()
wide_lik.columns.name = None
wide_lik["problem_number"] = wide_lik["problem_number"].astype(int)

OVERALL_MEANS = {
    "LQ": wide_lik["LQ"].mean(),
    "CA": wide_lik["CA"].mean(),
    "RP": wide_lik["RP"].mean(),
}
print(f"Overall means: LQ={OVERALL_MEANS['LQ']:.3f}  "
      f"CA={OVERALL_MEANS['CA']:.3f}  RP={OVERALL_MEANS['RP']:.3f}")

EXCLUDE = {
    "No Math Issue", "No grammar issue", "No cultural mistake",
    "Drop response - invalid", "Unclear or unspecified issue",
    "Did not understand instructions",
}

merged = pd.merge(
    tax_df[~tax_df["label"].isin(EXCLUDE)],
    wide_lik[["ResponseId", "problem_number", "model",
              "language_label", "resource_tier", "LQ", "CA", "RP"]],
    on=["ResponseId", "problem_number", "model"],
    how="inner",
)
print(f"Rows after merge and EXCLUDE filter: {len(merged)}")

# Deduplicate at (evaluation × label) level.
# A label tagged multiple times for the same (ResponseId, problem, model)
merged = merged.drop_duplicates(
    subset=["ResponseId", "problem_number", "model", "label"]
).reset_index(drop=True)
print(f"Rows after deduplication: {len(merged)}")

# Category -> Likert dimension mapping
CAT_DIM = {
    "Grammar Erros":  "LQ",
    "Culture Errors": "CA",
    "Math Errors":    "RP",
}

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
CAT_DIM = {"Grammar": "LQ", "Culture": "CA", "Math": "RP"}

merged["super_category"] = merged["label"].map(SUPER).fillna("Other")
merged

rng = np.random.default_rng(42)

def bootstrap_mean_ci(arr, n_boot=10000):
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return np.nan, np.nan
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])

MIN_N = 10

gap_rows = []
for (super_cat, label), grp in merged.groupby(["super_category", "label"]):
    if super_cat not in CAT_DIM:
        continue
    dim = CAT_DIM[super_cat]
    ratings = grp[dim].dropna().values
    if len(ratings) < MIN_N:
        continue
    mean_r = np.mean(ratings)
    ci_lo, ci_hi = bootstrap_mean_ci(ratings)
    overall_m = OVERALL_MEANS[dim]
    gap = overall_m - mean_r  # positive = mean below overall

    gap_rows.append({
        "super_category":   super_cat,
        "label":            label,
        "relevant_dim":     dim,
        "n":                len(ratings),
        "mean_rating":      round(mean_r, 3),
        "ci_lo":            round(ci_lo, 3),
        "ci_hi":            round(ci_hi, 3),
        "overall_mean":     round(overall_m, 3),
        "gap_from_overall": round(gap, 3),
    })

gap_df = pd.DataFrame(gap_rows).sort_values(
    ["super_category", "mean_rating"], ascending=[True, False]
)
gap_df.to_csv(f"{OUT_DIR}/detection_gap.csv", index=False)
print(f"Labels included (n >= {MIN_N}): {len(gap_df)}")

print(f"\n{'Category':<10} {'Label':<45} {'n':>5}  "
      f"{'mean':>6}  {'95% CI':>16}  {'gap':>6}")
print("-" * 95)
for cat in ["Grammar", "Culture", "Math"]:
    sub = gap_df[gap_df["super_category"] == cat].sort_values("mean_rating")
    overall = OVERALL_MEANS[CAT_DIM[cat]]
    print(f"\n  [{cat}]  overall mean {CAT_DIM[cat]} = {overall:.3f}")
    for _, r in sub.iterrows():
        print(f"  {r['super_category']:<10} {r['label']:<45} {r['n']:>5}  "
              f"{r['mean_rating']:>6.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
              f"{r['gap_from_overall']:>+6.3f}")

# Frequency counts: share of substantive label-instances (post-dedup).
freq_rows = []

# Overall
total_all = len(merged)
counts_all = (merged.groupby(["label", "super_category"])
                    .size().reset_index(name="count"))
counts_all["scope"] = "all"
counts_all["pct_of_obs"] = 100 * counts_all["count"] / total_all
freq_rows.append(counts_all)

# Per-language
for lang, sub in merged.groupby("language_label"):
    total_l = len(sub)
    c = (sub.groupby(["label", "super_category"])
            .size().reset_index(name="count"))
    c["scope"] = lang
    c["pct_of_obs"] = 100 * c["count"] / total_l
    freq_rows.append(c)

freq_df = pd.concat(freq_rows, ignore_index=True)
freq_df.to_csv(f"{OUT_DIR}/label_frequencies.csv", index=False)
print(f"freq_df rows: {len(freq_df)}")

COL_INVIS = "#1a7f8e"
COL_VIS   = "#c0392b"
COL_MID   = "#888888"
COL_REF   = "#3d3d3d"

COMPRESSIONS = [
    (" and ", " & "),
    (" or ",  " / "),
    ("Unspecified ", ""),
    ("errors", "err."),
    ("issue", "iss."),
    ("mismatch", "mismatch"),   # already short, leave
]

LABEL_OVERRIDES = {
    "Word choice and lexical mistranslation":    "Word choice / lexical mistranslation",
    "Fluency, awkwardness, and redundancy":      "Fluency & awkwardness",
    "Language complexity and age level":         "Language complexity & age",
    "Spelling, orthography, and diacritics":     "Spelling & diacritics",
    "Economic context: prices and affordability":"Economic context",
    "Language choice for local math convention": "Local math convention",
    "Attributed item from the wrong culture":    "Wrong-culture attribution",
    "Internal contradiction in scenario":        "Internal contradiction",
    "Unrealistic or implausible scenario":       "Unrealistic scenario",
}

def short_label(s, maxlen=32):
    s = LABEL_OVERRIDES.get(s, s)
    for old, new in COMPRESSIONS:
        s = s.replace(old, new)
    if len(s) <= maxlen:
        return s
    truncated = s[:maxlen]
    last_space = truncated.rfind(" ")
    if last_space > maxlen * 0.6:
        return truncated[:last_space] + "…"
    return truncated + "…"


def assign_color(gap):
    if gap >= 0.3:
        return COL_VIS
    elif gap <= 0.05:
        return COL_INVIS
    else:
        return COL_MID

PANEL_META = [
    {"cat": "Grammar", "dim": "LQ", "title": "Grammar errors"},
    {"cat": "Culture", "dim": "CA", "title": "Culture errors"},
    {"cat": "Math",    "dim": "RP", "title": "Math errors"},
]

#Per-panel data
panels = []
for meta in PANEL_META:
    sub = (gap_df[gap_df["super_category"] == meta["cat"]]
           .sort_values("mean_rating", ascending=True)
           .reset_index(drop=True))
    panels.append((meta, sub))

# Column widths proportional to label count so text isn't cramped
n_labels  = [len(sub) for _, sub in panels]
col_widths = [n * 0.30 + 0.8 for n in n_labels]   # inches per panel
total_w    = sum(col_widths)                         # should land ~6.73"

fig = plt.figure(figsize=(7.5, 3.6))
gs  = gridspec.GridSpec(
    1, 3,
    figure=fig,
    width_ratios=col_widths,
    wspace=0.85,
)

for col_idx, (meta, sub) in enumerate(panels):
    ax      = fig.add_subplot(gs[col_idx])
    dim     = meta["dim"]
    overall = OVERALL_MEANS[dim]

    for i, (_, row) in enumerate(sub.iterrows()):
        color = assign_color(row["gap_from_overall"])
        ax.plot(
            [row["ci_lo"], row["ci_hi"]], [i, i],
            color=color, linewidth=1.1,
            solid_capstyle="round", zorder=2,
        )
        ax.plot(
            row["mean_rating"], i,
            marker="o", color=color, markersize=4,
            markeredgecolor="white", markeredgewidth=0.35, zorder=3,
        )

    ax.axvline(
        x=overall, color=COL_REF,
        linewidth=0.8, linestyle="--", alpha=0.7, zorder=1,
    )

    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        [short_label(r["label"]) for _, r in sub.iterrows()],
        fontsize=6.5,
    )
    ax.set_ylim(-0.7, len(sub) - 0.3)

    all_vals = pd.concat([sub["ci_lo"], sub["ci_hi"]])
    xmin = min(all_vals.min(), overall) - 0.12
    xmax = max(all_vals.max(), overall) + 0.18
    ax.set_xlim(xmin, xmax)

    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("Mean rating (1–5)", fontsize=7, labelpad=3)
    ax.set_title(meta["title"], fontsize=8, fontweight="bold",
                 loc="left", pad=4)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.35, color="#e0e0e0", zorder=0)

fig.savefig(f"{OUT_DIR}/figure_detection_gap_triplet.pdf",
            bbox_inches="tight", format="pdf")
fig.savefig(f"{OUT_DIR}/figure_detection_gap_triplet.png",
            bbox_inches="tight", dpi=300)
plt.close()

TOP_N = 20
CAT_COLORS = {
    "Grammar": "#1a7f8e",
    "Culture": "#7b4fa6",
    "Math":    "#c0392b",
    "Other":   "#3d3d3d",
}

overall_top = (freq_df[freq_df["scope"] == "all"]
               .sort_values("count", ascending=False)
               .head(TOP_N)
               .sort_values("count", ascending=True)
               .reset_index(drop=True))

fig, ax = plt.subplots(figsize=(6.73, 5.2))
fig.subplots_adjust(left=0.48, right=0.97, top=0.92, bottom=0.12)

for i, (_, row) in enumerate(overall_top.iterrows()):
    color = CAT_COLORS.get(row["super_category"], "#888888")
    ax.barh(i, row["count"], color=color, alpha=0.85, height=0.7)
    ax.text(row["count"] + 2, i, f"{row['pct_of_obs']:.1f}%",
            va="center", fontsize=8.5, color="#444444")

ax.set_yticks(range(TOP_N))
ax.set_yticklabels([short_label(r["label"], 42)
                    for _, r in overall_top.iterrows()], fontsize=9)
ax.set_xlabel("Count (open-ended responses coded)", fontsize=9, labelpad=4)
ax.set_xlim(0, overall_top["count"].max() * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)

handles = [mpatches.Patch(color=CAT_COLORS[c], alpha=0.85, label=c)
           for c in ["Grammar", "Culture", "Math"]]
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=9)

plt.savefig(f"{OUT_DIR}/figure_frequencies.pdf", bbox_inches="tight", format="pdf")
plt.savefig(f"{OUT_DIR}/figure_frequencies.png", bbox_inches="tight", dpi=300)
plt.close()

# Per-language error frequency figures — one figure per language

LANGUAGES  = list(LANG_META.keys())
TOP_N_LANG = 15

for lang in LANGUAGES:
    lang_freq = (
        freq_df[freq_df["scope"] == lang]
        .sort_values("count", ascending=False)
        .head(TOP_N_LANG)
        .sort_values("count", ascending=True)
        .reset_index(drop=True)
    )

    n_items = len(lang_freq)
    fig_h   = max(2.8, n_items * 0.28 + 0.7)   # scale height to label count
    fig, ax = plt.subplots(figsize=(3.03, fig_h))
    fig.subplots_adjust(left=0.44, right=0.94, top=0.90, bottom=0.12)

    for i, (_, row) in enumerate(lang_freq.iterrows()):
        color = CAT_COLORS.get(row["super_category"], "#888888")
        ax.barh(i, row["count"], color=color, alpha=0.85, height=0.72)
        ax.text(
            row["count"] + max(lang_freq["count"].max() * 0.02, 0.5), i,
            f"{row['pct_of_obs']:.1f}%",
            va="center", fontsize=7.5, color="#444444",
        )

    tier   = LANG_META[lang]["tier"]
    region = LANG_META[lang]["region"]

    ax.set_yticks(range(n_items))
    ax.set_yticklabels(
        [short_label(r["label"], 30) for _, r in lang_freq.iterrows()],
        fontsize=7.5,
    )
    ax.set_title(
        f"{lang}  ·  {tier} resource  ·  {region}",
        fontsize=8, fontweight="bold", pad=5, loc="left",
    )
    ax.set_xlabel("Count (coded labels)", fontsize=8, labelpad=3)
    ax.set_xlim(0, lang_freq["count"].max() * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=7.5)

    handles = [
        mpatches.Patch(color=CAT_COLORS[c], alpha=0.85, label=c)
        for c in ["Grammar", "Culture", "Math"]
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=7.5)

    slug = lang.lower()
    fig.savefig(f"{OUT_DIR}/figure_frequencies_{slug}.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig(f"{OUT_DIR}/figure_frequencies_{slug}.png",
                bbox_inches="tight", dpi=300)
    plt.close()

# Total errors per language  (stacked bar + printed table)

lang_cat = (
    merged.groupby(["language_label", "super_category"])
    .size().reset_index(name="count")
)

pivot_totals = (
    lang_cat
    .pivot(index="language_label", columns="super_category", values="count")
    .fillna(0)
    .astype(int)
)
pivot_totals["TOTAL"] = pivot_totals.sum(axis=1)
pivot_totals = pivot_totals.sort_values("TOTAL", ascending=False)

grand = pivot_totals.sum()

# print table
cats_present = [c for c in ["Grammar", "Culture", "Math", "Other"]
                if c in pivot_totals.columns]
header = f"{'Language':<12}" + "".join(f"{c:>10}" for c in cats_present) + f"{'TOTAL':>10}"
print("\nTotal error labels by language (substantive, post-dedup)\n")
print(header)
print("-" * len(header))
for lang, row in pivot_totals.iterrows():
    vals = "".join(f"{int(row.get(c, 0)):>10}" for c in cats_present)
    print(f"{lang:<12}{vals}{int(row['TOTAL']):>10}")
print("-" * len(header))
grand_vals = "".join(f"{int(grand.get(c, 0)):>10}" for c in cats_present)
print(f"{'GRAND TOTAL':<12}{grand_vals}{int(grand['TOTAL']):>10}")

# stacked bar figure
langs_ord = pivot_totals.index.tolist()
bottoms   = np.zeros(len(langs_ord))

fig, ax = plt.subplots(figsize=(3.03, 2.8))
for cat in ["Grammar", "Culture", "Math"]:
    if cat not in pivot_totals.columns:
        continue
    vals = [pivot_totals.loc[l, cat] for l in langs_ord]
    ax.bar(langs_ord, vals, bottom=bottoms,
           color=CAT_COLORS[cat], alpha=0.85, label=cat)
    bottoms += np.array(vals, dtype=float)

# Label totals on top of each bar
for i, lang in enumerate(langs_ord):
    ax.text(i, bottoms[i] + 1, str(int(pivot_totals.loc[lang, "TOTAL"])),
            ha="center", va="bottom", fontsize=7, color="#333333")

ax.set_ylabel("Error count (coded labels)", fontsize=8)
ax.set_xticks(range(len(langs_ord))) # Added to fix UserWarning
ax.set_xticklabels(langs_ord, rotation=35, ha="right", fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=7.5)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/figure_total_errors_by_language.pdf",
            bbox_inches="tight", format="pdf")
fig.savefig(f"{OUT_DIR}/figure_total_errors_by_language.png",
            bbox_inches="tight", dpi=300)
plt.close()

# Worst-performing questions

BOTTOM_N = 5

#  per (problem × language)
q_lang = (
    wide_lik.groupby(["problem_number", "language_label"])
    .agg(
        mean_LQ=("LQ", "mean"),
        mean_CA=("CA", "mean"),
        mean_RP=("RP", "mean"),
        n_responses=("LQ", "count"),
    )
    .reset_index()
)
q_lang["composite"] = q_lang[["mean_LQ", "mean_CA", "mean_RP"]].mean(axis=1)

#  per problem (across all languages and models)
q_overall = (
    wide_lik.groupby("problem_number")
    .agg(
        mean_LQ=("LQ", "mean"),
        mean_CA=("CA", "mean"),
        mean_RP=("RP", "mean"),
        n_responses=("LQ", "count"),
    )
    .reset_index()
)
q_overall["composite"] = q_overall[["mean_LQ", "mean_CA", "mean_RP"]].mean(axis=1)

#  print: worst per language
print(f"\n{'=':=<70}")
print(f"WORST-PERFORMING QUESTIONS BY LANGUAGE  (bottom {BOTTOM_N} by composite mean)")
print(f"{'=':=<70}")
print(f"{'Language':<12} {'Q#':>4}  {'Composite':>10}  {'LQ':>6}  {'CA':>6}  {'RP':>6}  {'n':>4}")
print("-" * 60)

for lang in sorted(q_lang["language_label"].unique()):
    sub = (
        q_lang[q_lang["language_label"] == lang]
        .sort_values("composite")
        .head(BOTTOM_N)
    )
    for _, r in sub.iterrows():
        print(
            f"{lang:<12} {int(r['problem_number']):>4}  "
            f"{r['composite']:>10.3f}  "
            f"{r['mean_LQ']:>6.3f}  "
            f"{r['mean_CA']:>6.3f}  "
            f"{r['mean_RP']:>6.3f}  "
            f"{int(r['n_responses']):>4}"
        )
    print()

#  print: worst overall
print(f"\n{'=':=<70}")
print("WORST-PERFORMING QUESTIONS — ALL LANGUAGES COMBINED  (bottom 10)")
print(f"{'=':=<70}")
print(f"{'Q#':>4}  {'Composite':>10}  {'LQ':>6}  {'CA':>6}  {'RP':>6}  {'n':>5}")
print("-" * 45)
for _, r in q_overall.sort_values("composite").head(10).iterrows():
    print(
        f"{int(r['problem_number']):>4}  "
        f"{r['composite']:>10.3f}  "
        f"{r['mean_LQ']:>6.3f}  "
        f"{r['mean_CA']:>6.3f}  "
        f"{r['mean_RP']:>6.3f}  "
        f"{int(r['n_responses']):>5}"
    )

# save CSVs
q_lang.sort_values(["language_label", "composite"]).to_csv(
    f"{OUT_DIR}/worst_questions_by_language.csv", index=False
)
q_overall.sort_values("composite").to_csv(
    f"{OUT_DIR}/worst_questions_overall.csv", index=False
)
