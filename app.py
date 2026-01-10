import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ------------------ TITLE ------------------
st.title("🏠 Real Estate Price Fairness Checker")

st.write("Check whether a house listing is **Fair, Overpriced, or Underpriced** using ML.")

# ------------------ TRAIN DATA ------------------
data = {
    "Area": [1200, 1800, 1500, 1000, 2000],
    "BHK": [2, 3, 2, 1, 3],
    "Location": ["Bangalore", "Hyderabad", "Bangalore", "Chennai", "Hyderabad"],
    "Price": [4500000, 7500000, 6000000, 3500000, 8200000]
}

df = pd.DataFrame(data)

# Encode location
le = LabelEncoder()
df["Location_encoded"] = le.fit_transform(df["Location"])

# Train model
X = df[["Area", "BHK", "Location_encoded"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# ------------------ USER INPUT ------------------
st.header("Enter Property Details")

area = st.number_input("Area (sqft)", min_value=300, step=50)
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
location = st.selectbox("Location", le.classes_)
listed_price = st.number_input("Listed Price (₹)", min_value=100000)

# ------------------ PREDICTION ------------------
if st.button("Check Price Fairness"):
    location_encoded = le.transform([location])[0]

    sample = pd.DataFrame(
        [[area, bhk, location_encoded]],
        columns=["Area", "BHK", "Location_encoded"]
    )

    predicted_price = model.predict(sample)[0]

    upper = predicted_price * 1.10
    lower = predicted_price * 0.90

    if listed_price > upper:
        status = "🔴 Overpriced"
    elif listed_price < lower:
        status = "🔵 Underpriced"
    else:
        status = "🟢 Fair Price"

    st.subheader("Result")
    st.write(f"**Predicted Fair Price:** ₹ {int(predicted_price):,}")
    st.write(f"**Listed Price:** ₹ {listed_price:,}")
    st.write(f"**Status:** {status}")
