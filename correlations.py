# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs("results", exist_ok=True)

FILES = {
    "Bengali":  "bengali_evaluations.csv",
    "Italian":  "italian_evaluations.csv",
    "Hindi":    "hindi_evaluations.csv",
    "Urdu":     "urdu_evaluations.csv",
    "Punjabi":  "punjabi_evaluations.csv",
    "Sicilian": "sicilian_evaluations.csv",
    "Sindhi":   "sindhi_evaluations.csv",
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
likert["resource_tier"] = likert["language_label"].map({k: v["tier"] for k, v in LANG_META.items()})
likert["region"]        = likert["language_label"].map({k: v["region"] for k, v in LANG_META.items()})

wide = likert[likert["dimension"] != "unknown"].pivot_table(
    index=["ResponseId","problem_number","model","language_label","resource_tier","region"],
    columns="dimension", values="response_num"
).reset_index()
wide.columns.name = None
wide_clean = wide[["ResponseId","problem_number","model","language_label",
                   "resource_tier","region","LQ","CA","RP"]].dropna()

rng = np.random.default_rng(42)
def spearman_ci(x, y, n_boot=10000):
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    boots = [stats.spearmanr(x[idx := rng.choice(n, n, replace=True)],
                              y[idx])[0]
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, p, lo, hi

pairs = [("LQ","CA"), ("LQ","RP"), ("CA","RP")]

rows = []
scopes = (
    [("all", "all", wide_clean)]
    + [(t, "tier", wide_clean[wide_clean["resource_tier"]==t]) for t in ["high","mid","low"]]
    + [(r, "region", wide_clean[wide_clean["region"]==r]) for r in ["India","Italy","Pakistan"]]
    + [(l, "language", wide_clean[wide_clean["language_label"]==l])
       for l in ["Italian","Hindi","Bengali","Urdu","Punjabi","Sicilian","Sindhi"]]
)

for scope_val, scope_type, sub in scopes:
    for a, b in pairs:
        rho, p, lo, hi = spearman_ci(sub[a].values, sub[b].values)
        rows.append({
            "scope_type": scope_type,
            "scope":      scope_val,
            "n":          len(sub),
            "dim_a":      a,
            "dim_b":      b,
            "rho":        round(rho, 4),
            "ci_lo":      round(lo, 4),
            "ci_hi":      round(hi, 4),
            "p_value":    round(p, 6),
            "sig":        "*" if p < .05 else "",
        })
        print(f"  [{scope_type:<8} {scope_val:<10}] {a} x {b}: "
              f"rho={rho:.3f} [{lo:.3f}, {hi:.3f}]  p={p:.4f}")

pd.DataFrame(rows).to_csv("spearman_correlations.csv", index=False)
