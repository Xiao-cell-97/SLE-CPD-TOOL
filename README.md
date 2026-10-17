# SLE CPD Prediction Tool — English edition

Version: research-2.0.0-en. For research use only. Not for diagnosis or treatment decisions.

## Start locally
1. Extract the ZIP archive completely.
2. Open index.html in Edge, Chrome, Firefox, or Safari.
3. Select Model 1 (SLE vs healthy controls) or Model 2 (SLE vs non-SLE rheumatic diseases). Both include newly diagnosed, relapsing, and remitting SLE cases.
4. Enter all nine measurements in the units shown. Q-Flag requires the original analyzer value, not a binary flag.
5. Select Calculate probability. You may export the result as JSON.

No installation, server, network connection, or login is required. All scripts and model data are embedded in index.html. Mobile file-preview apps may not execute JavaScript; use a full browser or a published static webpage.

## Missing values and interpretation
All fields are required by default. Explicitly enable missing-value handling to use saved training fill values. Variables without a saved fill value use the original code's final zero fallback. Imputed values are listed in the result; all-blank submissions are rejected.

The output is the original model probability for the selected study comparison, without additional recalibration. It is not a diagnostic probability for a general clinical population. Thresholds and tertile cut-points are fixed from training-set nested-CV out-of-fold predictions. The probability groups have not been validated against independent clinical outcomes.

- Threshold comparison: p >= the locked Youden threshold.
- Low: p <= q1; Middle: q1 < p <= q2; High: p > q2.
- Model 1: threshold 0.6661904727899339; q1 0.6047152622937301; q2 0.8123788714824888.
- Model 2: threshold 0.6968560420622149; q1 0.4676059021936853; q2 0.7912400124435731.

EO, MONO and BASO are percentages; absolute cell counts use 10^9/L; MCHC uses g/L; MCV and RDW-SD use fL. Channel and distribution-width variables retain their original analyzer values. Device compatibility and intended-use eligibility require author review. Displayed training ranges are not clinical reference intervals.

## Provenance and validation
The supplied original final_model.pkl files were used to recover the LightGBM text models: 12 trees for Model 1 and 34 for Model 2. Browser inference uses the equivalent numeric trees. No model was retrained. Feature order, numerical parameters, imputation settings, thresholds and model hashes match the Chinese edition.

Numerical equivalence was checked against the original Python models using 3,170 synthetic inputs, including split boundaries. See QA.md. The English UI was checked using a simulated DOM; no actual browser rendering or interaction test has been performed. This is technical verification, not clinical validation.

## Files and publishing
- index.html: self-contained English webpage.
- inference.js / models.json: the same inference source and model data embedded in the page.
- models/: original LightGBM text models and settings.
- GITHUB_GUIDE.md: static hosting instructions.
- verification_results.json / QA.md: technical verification results and limitations.

The package contains no patient records, individual prediction tables, raw study data, or complete training/validation pipeline. Do not upload the original supplied model archive to a public repository: it also contains individual-level files and a spreadsheet.

The previous frontend's MIT license is retained in LICENSE. The authors must confirm model ownership, permissions and appropriate model licensing before public release; the frontend license does not establish model permissions. No GitHub repository or new live URL has been created by this delivery.

Inputs are processed in browser memory without network requests or persistent browser storage. Exported JSON files remain on the user's device and must be safeguarded.
