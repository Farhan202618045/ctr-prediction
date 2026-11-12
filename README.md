# Online Advertisement Click-Through Rate Prediction using LightGBM

Predicts the probability that an online ad impression gets clicked,
using a LightGBM model trained on the Criteo Click Logs dataset.

## Dataset
- Source: Criteo Click Logs (Hugging Face: criteo/CriteoClickLogs)
- Sample used: the first 25,000 rows of the training data
- 39 features: 13 numeric and 26 anonymised categorical
- 770 clicks out of 25,000 impressions (CTR 3.08%)

## Method
1. Stratified 80/20 train/test split
2. Rare categorical values (seen fewer than 20 times in training) grouped as "rare"
3. Missing values handled natively by LightGBM
4. Number of trees chosen by 5-fold cross-validation with early stopping
5. No class weighting, to keep the predicted probabilities calibrated

## Results on the test set (5,000 unseen ads)
| Metric | Model | Baseline |
|---|---|---|
| ROC-AUC | 0.690 | 0.500 |
| Log loss | 0.1309 | 0.1375 |
| Average predicted CTR | 3.03% | actual: 3.08% |

- The top 20% of ads by predicted probability had a 6.4% click rate,
  against 1.4% for the bottom 20%, and contained 41.6% of all clicks.
- Most important feature: integer_feature_4 (9.8% of total gain).

## Files
- `criteo_ctr_project.ipynb`: data loading, analysis, training and evaluation
- `app.py`: Streamlit app
- `ctr_model.pkl`: trained model and preprocessing information
- `sample_ads.csv`: 300 test ads used by the app
- `requirements.txt`, `packages.txt`: deployment settings

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
