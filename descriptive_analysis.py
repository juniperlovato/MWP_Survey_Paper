# -*- coding: utf-8 -*-

import os
import textwrap
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

BASE = "."

EVAL_DIR    = f"{BASE}/03 Cleaned Survey/Evaluation Dataset"
DESC_OUT    = f"{BASE}/Figures_Outputs/Descriptive_Analysis"
os.makedirs(DESC_OUT, exist_ok=True)

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

TIER_ORDER   = ["high", "mid", "low"]
REGION_ORDER = ["India", "Italy", "Pakistan"]

DIMS = ["linguistic_quality", "cultural_appropriateness", "reasoning_preservation"]
DIM_LABELS = {
    "linguistic_quality":       "Linguistic Quality",
    "cultural_appropriateness": "Cultural Appropriateness",
    "reasoning_preservation":   "Reasoning Preservation",
}

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

dfs = []
for lang, fpath in FILES.items():
    df = pd.read_csv(fpath)
    df["language_label"] = lang
    dfs.append(df)
    log(f"  Loaded {lang:<10}  rows={len(df):5d}  respondents={df['ResponseId'].nunique()}")

full_df = pd.concat(dfs, ignore_index=True)
log(f"\nTotal rows pooled : {len(full_df)}")
log(f"Total respondents : {full_df['ResponseId'].nunique()}")

likert = full_df[full_df["Question_No_Qualtrics"].str.endswith("_1", na=False)].copy()
likert["response_num"] = pd.to_numeric(likert["Response"], errors="coerce")

q = likert["Question_Qualtrics"].str.lower().fillna("")
likert["dimension"] = "unknown"
likert.loc[q.str.contains("grammar|natural flow"),            "dimension"] = "linguistic_quality"
likert.loc[q.str.contains("culturally appropriate"),          "dimension"] = "cultural_appropriateness"
likert.loc[q.str.contains("mathematical reasoning|preserve"), "dimension"] = "reasoning_preservation"

likert["resource_tier"] = likert["language_label"].map(
    {k: v["tier"] for k, v in LANG_META.items()}
)
likert["region"] = likert["language_label"].map(
    {k: v["region"] for k, v in LANG_META.items()}
)
likert["ambiguity"]  = pd.to_numeric(likert["Ambiguity "], errors="coerce").fillna(0).astype(int)
likert["complexity"] = pd.to_numeric(likert["No Elements"], errors="coerce")

log(f"\nLikert rows by dimension:")
for dim, n in likert["dimension"].value_counts().items():
    log(f"  {dim}: {n}")

log(f"\nLikert rows by language (post-exclusion respondent counts):")
for lang in sorted(LANG_META.keys()):
    sub = likert[likert["language_label"] == lang]
    tier = LANG_META[lang]["tier"]
    n_resp = sub["ResponseId"].nunique()
    n_rows = len(sub[sub["dimension"] != "unknown"]["response_num"].dropna())
    log(f"  {lang:<10} ({tier:<4})  respondents={n_resp:3d}  likert_obs={n_rows:5d}")

rng = np.random.default_rng(42)

def bootstrap_ci_mean(arr, n_boot=10000):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def bootstrap_median_diff_ci(a, b, n_boot=10000):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = [
        np.median(rng.choice(a, size=len(a), replace=True)) -
        np.median(rng.choice(b, size=len(b), replace=True))
        for _ in range(n_boot)
    ]
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def rank_biserial(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1 - (2 * u) / (nx * ny)

def epsilon_squared(h_stat, n, k):
    return (h_stat - k + 1) / (n - k)

def run_kw_and_mwu(sub_df, groupby_col, groups, dim, comp_label):
    rows = []
    gdata = {
        g: sub_df[sub_df[groupby_col] == g]["response_num"].dropna().values
        for g in groups
    }
    arrays = [gdata[g] for g in groups]
    n_total = sum(len(a) for a in arrays)

    h, p = stats.kruskal(*arrays)
    eps = epsilon_squared(h, n_total, len(groups))
    log(f"  KW: H={h:.3f}, p={p:.4f}, e2={eps:.4f}  (N={n_total})")
    rows.append({
        "dimension": dim, "comparison_type": comp_label,
        "test": "kruskal-wallis", "group_a": "all", "group_b": "",
        "n_a": n_total, "n_b": "",
        "statistic": round(h, 4), "p_value": round(p, 6),
        "effect_size": round(eps, 4), "effect_type": "epsilon_squared",
        "delta_median_ci_lo": "", "delta_median_ci_hi": "",
    })

    for g_a, g_b in combinations(groups, 2):
        a, b = gdata[g_a], gdata[g_b]
        u, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = rank_biserial(a, b)
        ci_lo, ci_hi = bootstrap_median_diff_ci(a, b)
        log(f"  MWU {g_a} vs {g_b}: U={u:.0f}, p={p_mwu:.4f}, r={rb:.3f}, "
            f"delta_median CI [{ci_lo:.3f}, {ci_hi:.3f}]  (n={len(a)}, {len(b)})")
        rows.append({
            "dimension": dim, "comparison_type": comp_label,
            "test": "mann-whitney-u", "group_a": g_a, "group_b": g_b,
            "n_a": len(a), "n_b": len(b),
            "statistic": round(u, 1), "p_value": round(p_mwu, 6),
            "effect_size": round(rb, 4), "effect_type": "rank_biserial_r",
            "delta_median_ci_lo": round(ci_lo, 4), "delta_median_ci_hi": round(ci_hi, 4),
        })
    return rows

log("DESCRIPTIVE DISTRIBUTIONS")

summary_rows = []

def add_summary(dim, group_var, group, vals):
    v = vals.dropna()
    ci_lo, ci_hi = bootstrap_ci_mean(v)
    summary_rows.append({
        "dimension": dim, "group_var": group_var, "group": group,
        "n": len(v), "mean": round(v.mean(), 4), "median": v.median(),
        "sd": round(v.std(), 4),
        "ci_lo_mean": round(ci_lo, 4), "ci_hi_mean": round(ci_hi, 4),
    })
    return ci_lo, ci_hi

for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    log(f"\n── {DIM_LABELS[dim]} ──")

    v = sub["response_num"]
    ci_lo, ci_hi = add_summary(dim, "overall", "all", v)
    log(f"  Overall  n={len(v.dropna())}  mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
        f"  median={v.median():.1f}  SD={v.std():.3f}")

    log("  By language:")
    for lang in sorted(LANG_META.keys()):
        v = sub[sub["language_label"] == lang]["response_num"]
        tier = LANG_META[lang]["tier"]
        ci_lo, ci_hi = add_summary(dim, "language", lang, v)
        log(f"    {lang:<10} ({tier:<4})  n={len(v.dropna()):4d}  "
            f"mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  "
            f"median={v.median():.1f}  SD={v.std():.3f}")

    log("  By resource tier:")
    for tier in TIER_ORDER:
        v = sub[sub["resource_tier"] == tier]["response_num"]
        ci_lo, ci_hi = add_summary(dim, "resource_tier", tier, v)
        log(f"    {tier:<6}  n={len(v.dropna()):5d}  mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
            f"  median={v.median():.1f}  SD={v.std():.3f}")

    log("  By region:")
    for region in REGION_ORDER:
        v = sub[sub["region"] == region]["response_num"]
        ci_lo, ci_hi = add_summary(dim, "region", region, v)
        log(f"    {region:<10}  n={len(v.dropna()):5d}  mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
            f"  median={v.median():.1f}  SD={v.std():.3f}")

    log("  By model:")
    for model, grp in sub.groupby("model"):
        v = grp["response_num"]
        ci_lo, ci_hi = add_summary(dim, "model", model, v)
        log(f"    {model:<20}  n={len(v.dropna()):5d}  mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
            f"  median={v.median():.1f}  SD={v.std():.3f}")

    log("  By ambiguity:")
    for amb, label in [(0, "standard"), (1, "ambiguous")]:
        v = sub[sub["ambiguity"] == amb]["response_num"]
        ci_lo, ci_hi = add_summary(dim, "ambiguity", label, v)
        log(f"    {label:<12}  n={len(v.dropna()):5d}  mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
            f"  median={v.median():.1f}  SD={v.std():.3f}")

    log("  By cultural complexity:")
    for cplx, grp in sub.groupby("complexity"):
        v = grp["response_num"]
        ci_lo, ci_hi = add_summary(dim, "complexity", f"level_{int(cplx)}", v)
        log(f"    Level {int(cplx)}  n={len(v.dropna()):5d}  mean={v.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
            f"  median={v.median():.1f}  SD={v.std():.3f}")

pd.DataFrame(summary_rows).to_csv(f"{DESC_OUT}/descriptive_summary.csv", index=False)

log("RESOURCE TIER COMPARISONS (H1 / H2 / H3)")

tier_rows = []
for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    log(f"\n── {DIM_LABELS[dim]} ──")
    tier_rows += run_kw_and_mwu(sub, "resource_tier", TIER_ORDER, dim, "resource_tier")

pd.DataFrame(tier_rows).to_csv(f"{DESC_OUT}/resource_tier_comparisons.csv", index=False)

log("REGION COMPARISONS (exploratory)")
log("Kruskal-Wallis (3 regions) + pairwise Mann-Whitney U")

region_rows = []
for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    log(f"\n── {DIM_LABELS[dim]} ──")
    region_rows += run_kw_and_mwu(sub, "region", REGION_ORDER, dim, "region")

pd.DataFrame(region_rows).to_csv(f"{DESC_OUT}/region_comparisons.csv", index=False)

log("WITHIN-TIER LANGUAGE VARIANCE (exploratory)")
log("Pairwise MWU between languages sharing the same resource tier.")
log("Tests whether tier-mates differ — if so, tier alone does not explain variance.")

tier_lang_map = {}
for lang, meta in LANG_META.items():
    tier_lang_map.setdefault(meta["tier"], []).append(lang)

wt_rows = []
for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    log(f"\n── {DIM_LABELS[dim]} ──")
    for tier in TIER_ORDER:
        langs = sorted(tier_lang_map[tier])
        if len(langs) < 2:
            continue
        log(f"  Tier: {tier}  languages: {langs}")
        for lang_a, lang_b in combinations(langs, 2):
            a = sub[sub["language_label"] == lang_a]["response_num"].dropna().values
            b = sub[sub["language_label"] == lang_b]["response_num"].dropna().values
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rb = rank_biserial(a, b)
            ci_lo, ci_hi = bootstrap_median_diff_ci(a, b)
            region_a = LANG_META[lang_a]["region"]
            region_b = LANG_META[lang_b]["region"]
            log(f"    {lang_a} ({region_a}) vs {lang_b} ({region_b}): "
                f"U={u:.0f}, p={p:.4f}, r={rb:.3f}, "
                f"delta_median CI [{ci_lo:.3f}, {ci_hi:.3f}]  (n={len(a)}, {len(b)})")
            wt_rows.append({
                "dimension": dim, "tier": tier,
                "lang_a": lang_a, "region_a": region_a,
                "lang_b": lang_b, "region_b": region_b,
                "same_region": region_a == region_b,
                "n_a": len(a), "n_b": len(b),
                "statistic": round(u, 1), "p_value": round(p, 6),
                "effect_size": round(rb, 4), "effect_type": "rank_biserial_r",
                "delta_median_ci_lo": round(ci_lo, 4),
                "delta_median_ci_hi": round(ci_hi, 4),
            })

wt_df = pd.DataFrame(wt_rows)
wt_df.to_csv(f"{DESC_OUT}/within_tier_language_variance.csv", index=False)

log(f"\n  Summary: within-tier pairs significant at p < .05")
for dim in DIMS:
    dim_wt = wt_df[wt_df["dimension"] == dim]
    n_pairs  = len(dim_wt)
    n_sig    = (dim_wt["p_value"] < 0.05).sum()
    n_cross  = (dim_wt["same_region"] == False).sum()
    n_cross_sig = ((dim_wt["p_value"] < 0.05) & (dim_wt["same_region"] == False)).sum()
    n_same_sig  = ((dim_wt["p_value"] < 0.05) & (dim_wt["same_region"] == True)).sum()
    log(f"  {DIM_LABELS[dim]}: {n_sig}/{n_pairs} pairs significant  "
        f"(cross-region: {n_cross_sig}/{n_cross}  same-region: {n_same_sig}/{n_pairs - n_cross})")

log("MODEL COMPARISONS (H4)")

models = sorted(likert["model"].unique())
model_rows = []
for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    log(f"\n── {DIM_LABELS[dim]} ──")
    model_rows += run_kw_and_mwu(sub, "model", models, dim, "model")

pd.DataFrame(model_rows).to_csv(f"{DESC_OUT}/model_comparisons.csv", index=False)

log("AMBIGUITY COMPARISONS (H5)")
log("Overall + broken out by resource tier")

amb_rows = []
for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    log(f"\n── {DIM_LABELS[dim]} ──")
    for scope_label, scope_sub in [("all", sub)] + [
        (tier, sub[sub["resource_tier"] == tier]) for tier in TIER_ORDER
    ]:
        g_std = scope_sub[scope_sub["ambiguity"] == 0]["response_num"].dropna().values
        g_amb = scope_sub[scope_sub["ambiguity"] == 1]["response_num"].dropna().values
        if len(g_std) == 0 or len(g_amb) == 0:
            continue
        u, p = stats.mannwhitneyu(g_std, g_amb, alternative="two-sided")
        rb = rank_biserial(g_std, g_amb)
        ci_lo, ci_hi = bootstrap_median_diff_ci(g_std, g_amb)
        log(f"  {scope_label:<6}: std (n={len(g_std):4d}) vs amb (n={len(g_amb):4d}): "
            f"U={u:.0f}, p={p:.4f}, r={rb:.3f}, delta_median CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        amb_rows.append({
            "dimension": dim, "scope": scope_label,
            "n_standard": len(g_std), "n_ambiguous": len(g_amb),
            "statistic": round(u, 1), "p_value": round(p, 6),
            "effect_size": round(rb, 4), "effect_type": "rank_biserial_r",
            "delta_median_ci_lo": round(ci_lo, 4), "delta_median_ci_hi": round(ci_hi, 4),
        })

pd.DataFrame(amb_rows).to_csv(f"{DESC_OUT}/ambiguity_comparisons.csv", index=False)

log("MISMATCH CHECKLIST FREQUENCIES")
log("Broken out by: overall, resource tier, region, language")

checklist_df = full_df[
    full_df["Question_Qualtrics"].str.contains("Which aspects|did not align", case=False, na=False)
].copy()
checklist_df["resource_tier"] = checklist_df["language_label"].map(
    {k: v["tier"] for k, v in LANG_META.items()}
)
checklist_df["region"] = checklist_df["language_label"].map(
    {k: v["region"] for k, v in LANG_META.items()}
)

def parse_checklist(response_str):
    r = str(response_str)
    return [
        label for label in sorted(CHECKLIST_LABELS, key=len, reverse=True)
        if label.rstrip(":") in r
    ]

freq_rows = []

scopes = (
    [("all", checklist_df)]
    + [(tier, checklist_df[checklist_df["resource_tier"] == tier]) for tier in TIER_ORDER]
    + [(region, checklist_df[checklist_df["region"] == region]) for region in REGION_ORDER]
    + [(lang, checklist_df[checklist_df["language_label"] == lang])
       for lang in sorted(LANG_META.keys())]
)

for scope_label, scope_df in scopes:
    total = len(scope_df["Response"].dropna())
    counts = Counter()
    for resp in scope_df["Response"].dropna():
        for label in parse_checklist(resp):
            counts[label] += 1

    log(f"\n  Scope: {scope_label}  (n responses = {total})")
    for label in CHECKLIST_LABELS:
        n = counts.get(label, 0)
        pct = 100 * n / total if total > 0 else 0
        log(f"    {n:4d} ({pct:5.1f}%)  {textwrap.shorten(label, 68, placeholder='...')}")
        freq_rows.append({
            "scope": scope_label, "category": label,
            "count": n, "total_responses": total,
            "pct_of_responses": round(pct, 2),
        })

pd.DataFrame(freq_rows).to_csv(f"{DESC_OUT}/checklist_frequencies.csv", index=False)

with open(f"{DESC_OUT}/analysis_log.txt", "w") as f:
    f.write("\n".join(log_lines))

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
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
})

# Compute per-language means with bootstrap CIs
fig_data = {dim: {} for dim in DIMS}
for dim in DIMS:
    sub = likert[likert["dimension"] == dim]
    for lang in LANG_META.keys():
        vals = sub[sub["language_label"] == lang]["response_num"].dropna().values
        if len(vals) == 0:
            continue
        mean   = float(np.mean(vals))
        ci_lo, ci_hi = bootstrap_ci_mean(vals)
        fig_data[dim][lang] = (mean, ci_lo, ci_hi)

# Region styling
REGION_STYLE = {
    "India":    {"color": "#2166AC", "marker": "o"},
    "Italy":    {"color": "#4DAC26", "marker": "s"},
    "Pakistan": {"color": "#D6604D", "marker": "^"},
}

TIER_POS    = {"high": 0, "mid": 1, "low": 2}
TIER_LABELS = ["High", "Mid", "Low"]

# Derive tier groupings from LANG_META, sorting for stable jitter assignment
TIER_LANGS_ORDERED = {tier: [] for tier in TIER_ORDER}
for lang, meta in LANG_META.items():
    TIER_LANGS_ORDERED[meta["tier"]].append(lang)
for tier in TIER_LANGS_ORDERED:
    TIER_LANGS_ORDERED[tier].sort()

# Assign even x-jitter within each tier
JITTER = {}
for tier, langs in TIER_LANGS_ORDERED.items():
    n = len(langs)
    if n == 1:
        offsets = [0.0]
    elif n == 2:
        offsets = [-0.12, 0.12]
    elif n == 3:
        offsets = [-0.18, 0.00, 0.18]
    else:
        offsets = np.linspace(-0.20, 0.20, n).tolist()
    JITTER[tier] = dict(zip(langs, offsets))

fig, axes = plt.subplots(1, 3, figsize=(6.73, 2.9), sharey=False)
fig.subplots_adjust(wspace=0.38)

for ax, dim in zip(axes, DIMS):
    dim_data = fig_data[dim]

    for lang, (mean, ci_lo, ci_hi) in dim_data.items():
        tier   = LANG_META[lang]["tier"]
        region = LANG_META[lang]["region"]
        x      = TIER_POS[tier] + JITTER[tier][lang]
        style  = REGION_STYLE[region]
        yerr   = [[mean - ci_lo], [ci_hi - mean]]

        ax.errorbar(
            x, mean, yerr=yerr,
            fmt=style["marker"],
            color=style["color"],
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=3,
            capthick=0.8,
            elinewidth=0.8,
            linewidth=0,
            zorder=3,
        )

        langs_in_tier = TIER_LANGS_ORDERED[tier]
        idx_in_tier = langs_in_tier.index(lang)
        va, off = ("bottom", 0.013) if idx_in_tier % 2 == 0 else ("top", -0.013)

        ax.annotate(
            lang,
            xy=(x, mean),
            xytext=(x, mean + off),
            fontsize=6.5,
            ha="center",
            va=va,
            color=style["color"],
        )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(TIER_LABELS)
    ax.set_xlabel("Resource Tier", labelpad=4)
    ax.set_title(DIM_LABELS[dim], pad=6)
    ax.set_xlim(-0.45, 2.45)

    all_cis = [ci_lo for _, ci_lo, _ in dim_data.values()]
    all_cih = [ci_hi for _, _, ci_hi in dim_data.values()]
    ymin = min(all_cis) - 0.04
    ymax = max(all_cih) + 0.06
    ax.set_ylim(ymin, ymax)

axes[0].set_ylabel("Mean Rating (1–5)", labelpad=4)

# legend
legend_handles = [
    Line2D([0], [0], marker=s['marker'], color=s['color'], label=r,
           markersize=6, linestyle='None',
           markeredgecolor='white', markeredgewidth=0.5)
    for r, s in REGION_STYLE.items()
]

fig.legend(
    handles=legend_handles,
    title="Region",
    loc='upper left',
    bbox_to_anchor=(1.0, 1.0),
    frameon=False,
    fontsize=8
)

plt.savefig(f"{DESC_OUT}/figure1_within_tier.pdf", bbox_inches="tight", format="pdf")
plt.savefig(f"{DESC_OUT}/figure1_within_tier.png", bbox_inches="tight", dpi=300)
plt.close()

CHECKLIST_CAT = {
    "Names of people, places, or objects did not reflect the target culture":              "Cultural Fit",
    "Settings or locations did not feel realistic for the region":                         "Cultural Fit",
    "Social values or norms (e.g., family roles, gender roles, authority) were not culturally appropriate": "Cultural Fit",
    "Age appropriateness - text was not aligned with expectations for school-aged children": "Cultural Fit",
    "Everyday practices (e.g., school system, sports, holidays, foods, religious references) were inaccurate or unfamiliar": "Cultural Fit",
    "Content that may be perceived as culturally insensitive or offensive":                "Cultural Fit",
    "The translation relied on generalized or overly simplistic cultural representations": "Cultural Fit",
    "The translation did not reflect the diversity or complexity of the culture":           "Cultural Fit",
    "Communication style (e.g., politeness, directness, tone) did not fit the cultural norms": "Language & Expression",
    "Idioms, metaphors, or expressions were confusing or inappropriate":                   "Language & Expression",
    "Humor was not culturally appropriate or understandable":                              "Language & Expression",
    "Phrases or vocabulary felt awkward or not typical of how people actually speak in the local context": "Language & Expression",
    "Units, currency, or assumptions (e.g., measurement systems, pricing) did not match the culture": "Math & Format",
    "Format or problem type does not resemble what is typically used in local math education": "Math & Format",
}

# Culture purple, Grammar teal, Math red
CHECKLIST_CAT_COLORS = {
    "Cultural Fit":          "#7b4fa6",
    "Language & Expression": "#1a7f8e",
    "Math & Format":         "#c0392b",
}

# Short display labels (for bar chart y-axis)
CHECKLIST_SHORT = {
    "Names of people, places, or objects did not reflect the target culture":              "Names/objects not culturally appropriate",
    "Settings or locations did not feel realistic for the region":                         "Settings/locations unrealistic",
    "Social values or norms (e.g., family roles, gender roles, authority) were not culturally appropriate": "Social norms inappropriate",
    "Age appropriateness - text was not aligned with expectations for school-aged children": "Age-appropriateness mismatch",
    "Everyday practices (e.g., school system, sports, holidays, foods, religious references) were inaccurate or unfamiliar": "Everyday practices inaccurate",
    "Communication style (e.g., politeness, directness, tone) did not fit the cultural norms": "Communication style mismatch",
    "Idioms, metaphors, or expressions were confusing or inappropriate":                   "Idioms/expressions confusing",
    "Humor was not culturally appropriate or understandable":                              "Humor not appropriate",
    "Units, currency, or assumptions (e.g., measurement systems, pricing) did not match the culture": "Units/currency mismatch",
    "Format or problem type does not resemble what is typically used in local math education": "Math format unfamiliar",
    "Content that may be perceived as culturally insensitive or offensive":                "Culturally insensitive content",
    "The translation relied on generalized or overly simplistic cultural representations": "Over-generalised representation",
    "The translation did not reflect the diversity or complexity of the culture":           "Cultural diversity not reflected",
    "Phrases or vocabulary felt awkward or not typical of how people actually speak in the local context": "Awkward phrasing/vocabulary",
    "None of the above, all aspects of the text felt culturally appropriate":              "None (all appropriate)",
    "Other (please specify):": "Other",
}

# Substantive labels only (excludes None/Other from frequency figures)
CHK_SUBSTANTIVE = [l for l in CHECKLIST_LABELS
                   if l not in {
                       "None of the above, all aspects of the text felt culturally appropriate",
                       "Other (please specify):",
                   }]

#  Long-form checklist DataFrame
# One row per (response × flagged label).
# Uses checklist_df

if "problem_number" not in checklist_df.columns:
    raise ValueError(
        "problem_number column missing from checklist_df. "
        "Check that evaluation CSVs include a problem_number column."
    )

chk_rows = []
for _, row in checklist_df.iterrows():
    flagged = parse_checklist(row["Response"])
    if not flagged:
        continue
    for label in flagged:
        chk_rows.append({
            "ResponseId":     row["ResponseId"],
            "problem_number": row["problem_number"],
            "model":          row.get("model", np.nan),
            "language_label": row["language_label"],
            "resource_tier":  row["resource_tier"],
            "region":         row["region"],
            "label":          label,
            "category":       CHECKLIST_CAT.get(label, "Other"),
        })

chk_long = pd.DataFrame(chk_rows)
chk_long["problem_number"] = pd.to_numeric(chk_long["problem_number"], errors="coerce")
print(f"chk_long rows (one per response×flag): {len(chk_long)}")
print(f"Unique problems: {chk_long['problem_number'].nunique()}")
print(f"Unique languages: {chk_long['language_label'].nunique()}")

# Checklist frequency figures

#  helper
def checklist_freq_table(scope_df):
    #Return DataFrame with count and pct_of_responses for each substantive label
    from collections import Counter
    total  = len(scope_df["Response"].dropna())
    counts = Counter()
    for resp in scope_df["Response"].dropna():
        for label in parse_checklist(resp):
            if label in CHK_SUBSTANTIVE:
                counts[label] += 1
    rows = []
    for label in CHK_SUBSTANTIVE:
        n = counts.get(label, 0)
        rows.append({
            "label":            label,
            "short_label":      CHECKLIST_SHORT.get(label, label[:40]),
            "category":         CHECKLIST_CAT.get(label, "Other"),
            "count":            n,
            "pct_of_responses": 100 * n / total if total > 0 else 0,
        })
    return pd.DataFrame(rows), total


#  Per-language figures — one file each
for lang in LANG_META.keys():
    scope_df         = checklist_df[checklist_df["language_label"] == lang]
    lang_freq, n_lang = checklist_freq_table(scope_df)
    lang_freq = lang_freq.sort_values("count", ascending=True).reset_index(drop=True)

    n_items = len(lang_freq)
    fig_h   = max(2.8, n_items * 0.28 + 0.7)
    fig, ax = plt.subplots(figsize=(3.03, fig_h))
    fig.subplots_adjust(left=0.52, right=0.65, top=0.90, bottom=0.12) # Adjusted right to make space

    for i, (_, row) in enumerate(lang_freq.iterrows()):
        color = CHECKLIST_CAT_COLORS.get(row["category"], "#888888")
        ax.barh(i, row["count"], color=color, alpha=0.85, height=0.72)
        if row["count"] > 0:
            ax.text(
                row["count"] + max(lang_freq["count"].max() * 0.02, 0.5), i,
                f"{row['pct_of_responses']:.1f}%",
                va="center",
                fontsize=7.5,
                color="#444444",
            )

    tier   = LANG_META[lang]["tier"]
    region = LANG_META[lang]["region"]

    ax.set_yticks(range(n_items))
    ax.set_yticklabels(lang_freq["short_label"], fontsize=7.5)
    ax.set_title(
        f"{lang}  ·  {tier} resource  ·  {region}",
        fontsize=8,
        fontweight="bold",
        pad=5,
        loc="left",
    )
    ax.set_xlabel("Count (responses flagging item)", fontsize=8, labelpad=3)
    ax.set_xlim(0, lang_freq["count"].max() * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=7.5)

    handles = [
        mpatches.Patch(color=CHECKLIST_CAT_COLORS[c], alpha=0.85, label=c)
        for c in ["Cultural Fit", "Language & Expression", "Math & Format"]
    ]

    ax.legend(handles=handles, bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0., frameon=False, fontsize=7.5)

    slug = lang.lower()
    fig.savefig(f"{DESC_OUT}/figure_checklist_frequencies_{slug}.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig(f"{DESC_OUT}/figure_checklist_frequencies_{slug}.png",
                bbox_inches="tight", dpi=300)
    plt.close()
