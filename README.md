#Materials for:

Lovato, J., & Suchdev, P. (2026). Beyond Language Resource Tier: A Survey Study of Culturally Situated Resource Scarcity in LLM-Generated Cultural Translation of Math Word Problems. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES 2026).

This repository contains the analysis code, supplementary tables/figures, and survey materials accompanying the paper. It serves as the official supplementary archive referenced in the camera-ready version.

Preregistration: OSF [STUDY00003740] ([(https://doi.org/10.17605/OSF.IO/4WE78)](https://doi.org/10.17605/OSF.IO/4WE78))
Paper: arXiv link forthcoming; proceedings link to be added upon publication
Contact: jlovato@uvm.edu

**Extended_version_of_manuscript_AIES.pdf**

Full extended version of the paper with supplementary materials 

**Surveys (folder)**

Bengali_India.pdf, Hindi_India.pdf, Italian_Italy.pdf, Punjabi_India.pdf, Sicilian_Italy.pdf, Sindhi_Pakistan.pdf, Urdu_Pakistan.pdf

All surveys implemented in qualtrics. For .qsf files please contact the authors

**Files (folder)**
bengali_evaluations.csv, hindi_evaluations.csv, italian_evaluations.csv, punjabi_evaluations.csv, sicilian_evaluations.csv, sindhi_evaluations.csvurdu_evaluations.csv

Survey response data, for access to the error taxonomy .json please contact the authors 

**descriptive_analysis.py** 

Inputs: per-language CSVs 

Figure outputs:

figure1_within_tier.pdf: Figure 1 (mean ratings by language, grouped by tier, region-coded markers with bootstrap CIs)
figure_checklist_frequencies_{language}.pdf (7 files): Figures 5, 6, 7 (per-language checklist mismatch profiles for Italy, Pakistan, India)

Table/stats outputs:

descriptive_summary.csv: Table 1 (mean/SD/median/CI by language, tier, region; also contains model, ambiguity, and complexity)
checklist_frequencies.csv: Table 5 (checklist frequencies by tier) 
resource_tier_comparisons.csv: Kruskal–Wallis and pairwise MWU results for H1–H3
region_comparisons.csv: exploratory region-level KW/MWU results
within_tier_language_variance.csv: the within-tier pairwise claims that motivate the region analysis
model_comparisons.csv: H4 null result
ambiguity_comparisons.csv: H5 null result (overall and per-tier)
analysis_log.txt: full console log of everything above

**regressions.py**

Inputs: same per-language evaluation CSVs as descriptive_analysis.py

Figure outputs:

figure_regressions_horizontal.pdf: Figure 2 (odds-ratio forest plot, confirmatory tier row + exploratory region row x three dimensions)

Table/stats outputs:

regression_confirmatory.csv: Table 6 (confirmatory ordinal logistic regressions, tier as primary predictor)
regression_exploratory.csv: Table 7 (exploratory ordinal regressions, region as primary predictor)
regression_mismatch.csv: Table 8 (logistic regressions predicting checklist mismatch presence, both specifications)
regression_log.txt: convergence diagnostics, log-likelihoods, AIC values, and McFadden psuedo-R squared. 

**convergence_analysis.py**

Inputs: the seven evaluation CSVs plus error_taxonomy_annotated.json (Label Studio export of the coded open-ended responses)

Figure outputs:

figure_worst_problems_profile.pdf: Figure 9 (convergence profile of the 15 worst problems: Likert composite, checklist flag rate, and taxonomy error rate as badness percentile ranks)

Table/stats outputs:

convergence_scores_overall.csv: underlying data for Figure 9
convergence_scores_by_language.csv: per-language convergence ranks

**errortaxonomy.py**

Inputs: the seven evaluation CSVs plus the annotated Label Studio JSON

Figure outputs:

figure_frequencies.pdf: Figure 3 (top 20 error labels by frequency, colored by supercategory)
figure_detection_gap_triplet.pdf: Figure 4 (detection gap panels for Grammar/Culture/Math with bootstrap CIs and dimension-mean reference lines)
figure_total_errors_by_language.pdf: Figure 8 (stacked taxonomy error counts by language)
figure_frequencies_{language}.pdf: per-language taxonomy profiles

Table/stats outputs:

detection_gap.csv: Table 4 (all labels with n, mean, CI, delta) and the data behind Figure 4
label_frequencies.csv: Figure 3 
worst_questions_by_language.csv / worst_questions_overall.csv: supporting data for the worst-problem discussion 

**correlations.py**

Inputs: the seven evaluation CSVs

Outputs:

spearman_correlations.csv: Table 3 in full (pairwise Spearman for LQ–CA, LQ–RP, CA–RP with 95% bootstrap CIs, disaggregated overall, by tier, by region, and by language). 
