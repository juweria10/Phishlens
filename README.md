# PhishLens: Streamlit UI

AI-based phishing URL detection (UCI PhiUSIIL dataset, Logistic Regression / Decision Tree / Random Forest).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Keep the folders next to `app.py`: `saved_model/`, `ML_results/`, `EDA_images/`, `assets/`.

## Pages
- **Scan**: one URL or a batch (up to 200), verdict, probability, risk scale, URL breakdown, top drivers, URL checks, CSV export
- **Overview**: scan activity, verdict split, dataset and model snapshot, EDA figures
- **Model performance**: scores by feature setting, confusion matrices, feature reliance, training plots
- **History**: searchable, filterable, exportable; kept per visitor in the browser session (private to each visitor; cleared on refresh; CSV export to keep a copy)

## How scanning works
The full model needs 50 features, 29 of which come from the page itself, so a typed URL cannot feed it.
The scanner therefore uses `phishing_model_url_only_pipeline.joblib` (21 features computed from the URL string in `phishlens/engine.py`).
`TLDLegitimateProb` and `URLCharProb` come from lookup tables built from your cleaned dataset (`python tools/build_assets.py <csv>`).
On the dataset URLs the rebuilt features reproduce the labels about 99.8% of the time.

## Known limit (worth stating in the report)
Every legitimate URL in the training data is a bare `https://www.` homepage. The model learned that pattern, so deep links
(`github.com/user/repo`) or hosts without `www` can score as phishing. The app marks those scans "Low reliability".
Fixing this needs more varied legitimate URLs in training.
