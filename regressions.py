# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from scipy import stats
import os, warnings
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

os.makedirs("results", exist_ok=True)
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

BASE = "filepath"
EVAL_DIR    = f"{BASE}/filepath"

# Load data
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
full_df = pd.concat(dfs, ignore_index=True)

# Build Likert dataframe
likert = full_df[full_df["Question_No_Qualtrics"].str.endswith("_1", na=False)].copy()
likert["response_num"] = pd.to_numeric(likert["Response"], errors="coerce")
q = likert["Question_Qualtrics"].str.lower().fillna("")
likert["dimension"] = "unknown"
likert.loc[q.str.contains("grammar|natural flow"),            "dimension"] = "linguistic_quality"
likert.loc[q.str.contains("culturally appropriate"),          "dimension"] = "cultural_appropriateness"
likert.loc[q.str.contains("mathematical reasoning|preserve"), "dimension"] = "reasoning_preservation"
likert["resource_tier"] = likert["language_label"].map({k: v["tier"] for k, v in LANG_META.items()})
likert["region"]        = likert["language_label"].map({k: v["region"] for k, v in LANG_META.items()})
likert["ambiguity"]     = pd.to_numeric(likert["Ambiguity "], errors="coerce").fillna(0).astype(int)
likert["complexity"]    = pd.to_numeric(likert["No Elements"], errors="coerce")

# Build mismatch presence dataframe
checklist_df = full_df[
    full_df["Question_Qualtrics"].str.contains("Which aspects|did not align", case=False, na=False)
].copy()
checklist_df["resource_tier"] = checklist_df["language_label"].map({k: v["tier"] for k, v in LANG_META.items()})
checklist_df["region"]        = checklist_df["language_label"].map({k: v["region"] for k, v in LANG_META.items()})
checklist_df["ambiguity"]     = pd.to_numeric(checklist_df["Ambiguity "], errors="coerce").fillna(0).astype(int)
checklist_df["complexity"]    = pd.to_numeric(checklist_df["No Elements"], errors="coerce")

def has_mismatch(resp):
    #1 if any mismatch selected (excluding None of the above), 0 otherwise.
    r = str(resp)
    if "None of the above" in r:
        return 0
    for label in CHECKLIST_LABELS[:-2]:  # exclude None and Other
        if label.rstrip(":") in r:
            return 1
    return 0

checklist_df["mismatch_present"] = checklist_df["Response"].apply(has_mismatch)

DIMS = ["linguistic_quality", "cultural_appropriateness", "reasoning_preservation"]
DIM_LABELS = {
    "linguistic_quality":       "Linguistic Quality",
    "cultural_appropriateness": "Cultural Appropriateness",
    "reasoning_preservation":   "Reasoning Preservation",
}

def prep_ordinal(sub, group_var, ref_map):
    #Build design matrix for ordinal logistic regression.
    needed = list(dict.fromkeys(
        ["response_num", group_var, "complexity", "ambiguity", "model",
         "resource_tier", "region"]
    ))
    df = sub[needed].dropna().copy()
    df["response_ord"] = pd.Categorical(
        df["response_num"].astype(int),
        categories=[1, 2, 3, 4, 5],
        ordered=True
    )
    # Dummy code group_var
    cats = [c for c in ref_map["cats"] if c != ref_map["ref"]]
    for cat in cats:
        df[f"{group_var}_{cat}"] = (df[group_var] == cat).astype(int)
    # Dummy code model (ref: gemini-2.5-pro, per H4 directional prediction)
    for m in ["claude-opus-4", "gpt-4.1"]:
        df[f"model_{m}"] = (df["model"] == m).astype(int)
    # Interaction: ambiguity × group_var dummies
    for cat in cats:
        df[f"amb_x_{group_var}_{cat}"] = df["ambiguity"] * df[f"{group_var}_{cat}"]
    return df, cats

def run_ordinal(df, cats, group_var, label):
    #Fit ordinal logistic regression and return results DataFrame
    group_dummies   = [f"{group_var}_{c}" for c in cats]
    model_dummies   = ["model_claude-opus-4", "model_gpt-4.1"]
    interaction_cols = [f"amb_x_{group_var}_{c}" for c in cats]
    exog_cols = group_dummies + ["complexity", "ambiguity"] + model_dummies + interaction_cols
    exog = df[exog_cols].astype(float)
    endog = df["response_ord"]
    try:
        mod = OrderedModel(endog, exog, distr="logit")
        res = mod.fit(method="bfgs", disp=False, maxiter=500)
        rows = []
        for var in exog_cols:
            coef = res.params[var]
            se   = res.bse[var]
            z    = res.tvalues[var]
            p    = res.pvalues[var]
            OR   = np.exp(coef)
            ci_lo = np.exp(coef - 1.96 * se)
            ci_hi = np.exp(coef + 1.96 * se)
            rows.append({
                "model_spec": label,
                "predictor": var,
                "coef": round(coef, 4),
                "OR": round(OR, 4),
                "CI_lo": round(ci_lo, 4),
                "CI_hi": round(ci_hi, 4),
                "z": round(z, 4),
                "p_value": round(p, 6),
                "sig": "*" if p < .05 else ("." if p < .1 else ""),
            })
        log(f"  Converged: {res.mle_retvals['converged']}  "
            f"  Log-lik: {res.llf:.2f}  AIC: {res.aic:.2f}")
        return pd.DataFrame(rows), res
    except Exception as e:
        log(f"  ERROR: {e}")
        return pd.DataFrame(), None

def run_logistic(df, group_var, cats, label, outcome_col):
    #Fit binary logistic regression and return results DataFrame.
    group_dummies    = [f"{group_var}_{c}" for c in cats]
    model_dummies    = ["model_claude-opus-4", "model_gpt-4.1"]
    interaction_cols = [f"amb_x_{group_var}_{c}" for c in cats]
    # Rebuild dummies on this df
    for cat in cats:
        df[f"{group_var}_{cat}"] = (df[group_var] == cat).astype(int)
    for m in ["claude-opus-4", "gpt-4.1"]:
        df[f"model_{m}"] = (df["model"] == m).astype(int)
    for cat in cats:
        df[f"amb_x_{group_var}_{cat}"] = df["ambiguity"] * df[f"{group_var}_{cat}"]
    exog_cols = group_dummies + ["complexity", "ambiguity"] + model_dummies + interaction_cols
    X = sm.add_constant(df[exog_cols].astype(float))
    y = df[outcome_col]
    try:
        mod = sm.Logit(y, X)
        res = mod.fit(disp=False, maxiter=200)
        rows = []
        for var in exog_cols:
            coef  = res.params[var]
            se    = res.bse[var]
            z     = res.tvalues[var]
            p     = res.pvalues[var]
            OR    = np.exp(coef)
            ci_lo = np.exp(coef - 1.96 * se)
            ci_hi = np.exp(coef + 1.96 * se)
            rows.append({
                "model_spec": label,
                "predictor": var,
                "coef": round(coef, 4),
                "OR": round(OR, 4),
                "CI_lo": round(ci_lo, 4),
                "CI_hi": round(ci_hi, 4),
                "z": round(z, 4),
                "p_value": round(p, 6),
                "sig": "*" if p < .05 else ("." if p < .1 else ""),
            })
        log(f"  Converged: {res.mle_retvals['converged']}  "
            f"  Log-lik: {res.llf:.2f}  AIC: {res.aic:.2f}  "
            f"  Pseudo-R2: {res.prsquared:.4f}")
        return pd.DataFrame(rows), res
    except Exception as e:
        log(f"  ERROR: {e}")
        return pd.DataFrame(), None

def print_results(res_df, title):
    log(f"\n  {title}")
    log(f"  {'Predictor':<40} {'OR':>7} {'[95% CI]':>18}  {'p':>8}  sig")
    log(f"  {'-'*82}")
    for _, row in res_df.iterrows():
        ci = f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]"
        log(f"  {row['predictor']:<40} {row['OR']:>7.3f} {ci:>18}  {row['p_value']:>8.4f}  {row['sig']}")

#Confirmatory models (resource tier)

TIER_REF = {"cats": ["high", "mid", "low"], "ref": "high"}
conf_rows = []

for dim in DIMS:
    sub = likert[likert["dimension"] == dim].copy()
    df_ord, cats = prep_ordinal(sub, "resource_tier", TIER_REF)
    log(f"\n── {DIM_LABELS[dim]}  (analytical n={len(df_ord)}) ──")
    res_df, fit = run_ordinal(df_ord, cats, "resource_tier",
                              f"confirmatory_{dim}")
    if not res_df.empty:
        res_df["dimension"] = dim
        conf_rows.append(res_df)
        print_results(res_df, f"Confirmatory — {DIM_LABELS[dim]}")

# Exploratory models (region)
log("EXPLORATORY: ORDINAL LOGISTIC REGRESSION (REGION)")
log("Outcome: Likert rating (1–5 ordinal)")
log("Predictors: region (ref=India) + complexity + ambiguity + model (ref=gemini-2.5-pro)")
log("            + ambiguity × region interaction")

REGION_REF = {"cats": ["India", "Italy", "Pakistan"], "ref": "India"}
expl_rows = []

for dim in DIMS:
    sub = likert[likert["dimension"] == dim].copy()
    log(f"\n── {DIM_LABELS[dim]}  (n={len(sub.dropna())}) ──")
    df_ord, cats = prep_ordinal(sub, "region", REGION_REF)
    res_df, fit  = run_ordinal(df_ord, cats, "region",
                               f"exploratory_{dim}")
    if not res_df.empty:
        res_df["dimension"] = dim
        expl_rows.append(res_df)
        print_results(res_df, f"Exploratory — {DIM_LABELS[dim]}")

# Mismatch presence (logistic)
log("LOGISTIC REGRESSION: MISMATCH PRESENCE")
log("Outcome: any cultural mismatch flagged (binary)")

mismatch_rows = []
cl_df = checklist_df.dropna(subset=["complexity", "ambiguity", "model",
                                     "resource_tier", "region",
                                     "mismatch_present"]).copy()

for spec_label, group_var, ref_map in [
    ("confirmatory_mismatch", "resource_tier", TIER_REF),
    ("exploratory_mismatch",  "region",        REGION_REF),
]:
    cats = [c for c in ref_map["cats"] if c != ref_map["ref"]]
    log(f"\n── {spec_label}  (n={len(cl_df)}) ──")
    res_df, fit = run_logistic(cl_df.copy(), group_var, cats,
                                spec_label, "mismatch_present")
    if not res_df.empty:
        mismatch_rows.append(res_df)
        print_results(res_df, spec_label)

#Save outputs
if conf_rows:
    pd.concat(conf_rows).to_csv("regression_confirmatory.csv", index=False)
if expl_rows:
    pd.concat(expl_rows).to_csv("regression_exploratory.csv", index=False)
if mismatch_rows:
    pd.concat(mismatch_rows).to_csv("regression_mismatch.csv", index=False)

with open("regression_log.txt", "w") as f:
    f.write("\n".join(log_lines))

"""#Figures"""

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
})

#  Load results
conf = pd.read_csv("regression_confirmatory.csv")
expl = pd.read_csv("regression_exploratory.csv")

DIMS = ["linguistic_quality", "cultural_appropriateness", "reasoning_preservation"]
DIM_LABELS = {
    "linguistic_quality":       "Linguistic\nQuality",
    "cultural_appropriateness": "Cultural\nAppropriateness",
    "reasoning_preservation":   "Reasoning\nPreservation",
}

# Predictor display labels and ordering
CONF_PREDICTORS = {
    "resource_tier_mid":            "Mid tier",
    "resource_tier_low":            "Low tier",
    "complexity":                   "Complexity",
    "ambiguity":                    "Ambiguity",
    "model_claude-opus-4":          "Claude",
    "model_gpt-4.1":                "GPT-4.1",
    "amb_x_resource_tier_mid":      "Amb × Mid",
    "amb_x_resource_tier_low":      "Amb × Low",
}
EXPL_PREDICTORS = {
    "region_Italy":                 "Italy",
    "region_Pakistan":              "Pakistan",
    "complexity":                   "Complexity",
    "ambiguity":                    "Ambiguity",
    "model_claude-opus-4":          "Claude",
    "model_gpt-4.1":                "GPT-4.1",
    "amb_x_region_Italy":           "Amb × Italy",
    "amb_x_region_Pakistan":        "Amb × Pakistan",
}

# Significant: dark navy; non-significant: medium grey
COLOR_SIG   = "#1A3A6B"   # dark blue  — significant
COLOR_NS    = "#999999"   # grey       — non-significant
COLOR_LINE  = "#666666"

def get_plot_data(df_results, dim, pred_map):
    rows = []
    dim_df = df_results[df_results["dimension"] == dim]
    for pred_key, pred_label in pred_map.items():
        row = dim_df[dim_df["predictor"] == pred_key]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append({
            "label":   pred_label,
            "OR":      r["OR"],
            "CI_lo":   r["CI_lo"],
            "CI_hi":   r["CI_hi"],
            "p":       r["p_value"],
            "sig":     r["p_value"] < 0.05,
        })
    return rows

# Figure
# 2 rows (confirmatory | exploratory) × 3 columns (dimensions: LQ, CA, RP)
fig, axes = plt.subplots(
    2, 3,
    figsize=(6.73, 5.6),
    gridspec_kw={"wspace": 0.12, "hspace": 0.35},
)

row_specs = [
    ("Confirmatory\n(Resource Tier)", CONF_PREDICTORS, conf),
    ("Exploratory\n(Region)",         EXPL_PREDICTORS, expl),
]

for row_idx, (row_title, pred_map, df_res) in enumerate(row_specs):
    for col_idx, dim in enumerate(DIMS):
        ax = axes[row_idx, col_idx]
        data = get_plot_data(df_res, dim, pred_map)
        if not data:
            ax.set_visible(False)
            continue

        y_pos = list(range(len(data)))[::-1]  # top to bottom

        for i, (y, d) in enumerate(zip(y_pos, data)):
            color = COLOR_SIG if d["sig"] else COLOR_NS
            marker = "D" if d["sig"] else "o"
            ms = 5 if d["sig"] else 4

            # CI line
            ax.plot(
                [d["CI_lo"], d["CI_hi"]], [y, y],
                color=color, linewidth=1.2, solid_capstyle="round", zorder=2,
            )
            # Point estimate
            ax.plot(
                d["OR"], y,
                marker=marker, color=color,
                markersize=ms, markeredgecolor="white",
                markeredgewidth=0.4, zorder=3,
            )
            # p-value annotation for significant results
            if d["sig"]:
                ax.annotate(
                    f"p={d['p']:.3f}",
                    xy=(d["CI_hi"], y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=7.5,
                    va="center",
                    color=COLOR_SIG,
                )

        # Reference line at OR=1
        ax.axvline(x=1.0, color="black", linewidth=0.8,
                   linestyle="--", alpha=0.5, zorder=1)

        # Y-axis labels — only on the leftmost column
        # (predictors are identical across the three dimensions in a row)
        ax.set_yticks(y_pos)
        if col_idx == 0:
            ax.set_yticklabels([d["label"] for d in data], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.7, len(data) - 0.3)

        # X-axis — extra room on right for p-value annotations
        xmax = max(d["CI_hi"] for d in data)
        xmin = min(d["CI_lo"] for d in data)
        pad  = (xmax - xmin) * 0.12
        ax.set_xlim(max(0.35, xmin - pad), xmax + pad + 0.35)

        # X label only on bottom row
        if row_idx == len(row_specs) - 1:
            ax.set_xlabel("Odds Ratio", fontsize=9, labelpad=4)

        # Column titles (dimensions) on top row only
        if row_idx == 0:
            ax.set_title(DIM_LABELS[dim], fontsize=9, fontweight="bold", pad=6)

        # Row label (confirmatory/exploratory) on left column only
        if col_idx == 0:
            ax.set_ylabel(row_title, fontsize=9, fontweight="bold", labelpad=8)

sig_handle = mlines.Line2D([], [], color=COLOR_SIG, marker="D",
                            markersize=5, markeredgecolor="white",
                            markeredgewidth=0.4, linewidth=1.2,
                            label="p < .05")
ns_handle  = mlines.Line2D([], [], color=COLOR_NS, marker="o",
                            markersize=4, markeredgecolor="white",
                            markeredgewidth=0.4, linewidth=1.2,
                            label="p \u2265 .05")
ref_handle = mlines.Line2D([], [], color="black", linestyle="--",
                            linewidth=0.8, alpha=0.5, label="OR = 1 (reference)")


fig.subplots_adjust(bottom=0.09, top=0.95, left=0.22)

plt.savefig(
    "figure_regressions_horizontal.pdf",
    bbox_inches="tight",
    format="pdf",
)
plt.savefig(
    "figure_regressions_horizontal.png",
    bbox_inches="tight",
    dpi=300,
)
