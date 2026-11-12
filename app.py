import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CTR Prediction", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load("ctr_model.pkl")


@st.cache_data
def load_ads(cat_cols):
    return pd.read_csv("sample_ads.csv", dtype={c: str for c in cat_cols})


bundle = load_bundle()
model = bundle["model"]
keep = bundle["keep"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
base_rate = bundle["base_rate"]
threshold = bundle["threshold"]
ads = load_ads(tuple(cat_cols))


def prepare(data):
    data = data.copy()
    for c in cat_cols:
        known = data[c].isin(keep[c]) | data[c].isnull()
        col = data[c].where(known, "rare")
        data[c] = pd.Categorical(col, categories=keep[c] + ["rare"])
    for c in num_cols:
        data[c] = data[c].astype(float)
    return data[num_cols + cat_cols]


def predict(row):
    return float(model.predict(prepare(row))[0])


st.title("Ad Click-Through Rate Prediction")
st.write(
    "A LightGBM model trained on 20,000 Criteo ad impressions. "
    "Pick a real ad the model has never seen, and it estimates "
    "the probability that the ad gets clicked."
)

ad_no = st.number_input("Choose an ad (1 to {})".format(len(ads)), 1, len(ads), 1)
ad = ads.iloc[[ad_no - 1]].reset_index(drop=True)
actual = int(ad.loc[0, "label"])
row = ad.drop(columns="label")
original_prob = predict(row)

st.subheader("Try changing the most important numeric features")
st.caption("Leave a box empty to treat the value as missing.")
edit_cols = [f for f in bundle["top_features"] if f in num_cols][:6]
boxes = st.columns(3)
for i, f in enumerate(edit_cols):
    current = row.loc[0, f]
    start = None if pd.isnull(current) else float(current)
    new = boxes[i % 3].number_input(f, value=start, step=1.0, key="{}_{}".format(f, ad_no))
    row[f] = np.nan if new is None else new

prob = predict(row)

st.subheader("Prediction")
a, b, c = st.columns(3)
a.metric("Click probability", "{:.2f}%".format(prob * 100))
b.metric("Average ad", "{:.2f}%".format(base_rate * 100))
c.metric("Compared to average", "{:.1f}x".format(prob / base_rate))

if prob >= threshold:
    st.success("Likely click: this ad is in the top 20% by predicted probability.")
else:
    st.info("Unlikely click: this ad is below the top-20% cut-off ({:.2f}%).".format(threshold * 100))

if abs(prob - original_prob) > 1e-9:
    st.write("With the original values the prediction was {:.2f}%.".format(original_prob * 100))

st.write("What really happened to this ad: **{}**".format("Clicked" if actual == 1 else "Not clicked"))

with st.expander("Show all 39 feature values of this ad"):
    st.dataframe(row.T.astype(str).rename(columns={0: "value"}))

st.caption(
    "The model outputs a probability, not a certainty. "
    "Features are anonymised by Criteo, so their real-world meaning is unknown."
)
