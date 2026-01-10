import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ------------------ DATA WITH LOCATION ------------------
data = {
    "Area": [1200, 1800, 1500, 1000, 2000],
    "BHK": [2, 3, 2, 1, 3],
    "Location": ["Bangalore", "Hyderabad", "Bangalore", "Chennai", "Hyderabad"],
    "Price": [4500000, 7500000, 6000000, 3500000, 8200000]
}

df = pd.DataFrame(data)

# ------------------ ENCODE LOCATION ------------------
le = LabelEncoder()
df["Location_encoded"] = le.fit_transform(df["Location"])

print("Location Encoding:")
print(df[["Location", "Location_encoded"]])

# ------------------ FEATURES & TARGET ------------------
X = df[["Area", "BHK", "Location_encoded"]]
y = df["Price"]

# ------------------ TRAIN MODEL ------------------
model = LinearRegression()
model.fit(X, y)

# ------------------ USER LISTING ------------------
listed_price = 6200000
area = 1400
bhk = 2
location = "Bangalore"

location_encoded = le.transform([location])[0]

sample = pd.DataFrame(
    [[area, bhk, location_encoded]],
    columns=["Area", "BHK", "Location_encoded"]
)

predicted_price = model.predict(sample)[0]

# ------------------ FAIRNESS LOGIC ------------------
upper = predicted_price * 1.10
lower = predicted_price * 0.90

if listed_price > upper:
    status = "🔴 Overpriced"
elif listed_price < lower:
    status = "🔵 Underpriced"
else:
    status = "🟢 Fair Price"

# ------------------ OUTPUT ------------------
print("\n--- Price Analysis ---")
print(f"Location               : {location}")
print(f"Predicted Fair Price   : ₹ {int(predicted_price):,}")
print(f"Listed Price           : ₹ {listed_price:,}")
print(f"Price Status           : {status}")
