# MWP_Survey_Paper
Repo for AIES/AAAI Beyond the Language Tier: A Survey Study of Culturally Situated Resource Scarcity in Math Word Problem Adaptation

descriptive_analysis.py 

Inputs: per-language cleaned evaluation CSVs (03 Cleaned Survey/Evaluation Dataset/{language}_evaluations.csv)

Figure outputs:

figure1_within_tier.pdf/.png → Figure 1 (mean ratings by language, grouped by tier, region-coded markers with bootstrap CIs)
figure_checklist_frequencies_{language}.pdf/.png (7 files) → Figures 5, 6, 7 (per-language checklist mismatch profiles for Italy, Pakistan, India)

Table/stats outputs:

descriptive_summary.csv → Table 1 (mean/SD/median/CI by language, tier, region; also contains model, ambiguity, and complexity breakdowns beyond what's tabled)
checklist_frequencies.csv → Table 5 (checklist frequencies by tier) 
resource_tier_comparisons.csv → Kruskal–Wallis and pairwise MWU results for H1–H3 (LQ H = 9.58, CA H = 14.62, RP H = 1.53)
region_comparisons.csv → exploratory region-level KW/MWU results
within_tier_language_variance.csv → the within-tier pairwise claims (Punjabi > Urdu, Sindhi < Sicilian, Bengali > Hindi) that motivate the region analysis
model_comparisons.csv → H4 null result
ambiguity_comparisons.csv → H5 null result (overall and per-tier)
analysis_log.txt → full console log of everything above
