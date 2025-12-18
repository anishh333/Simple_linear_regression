import streamlit as slt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# page configuration
slt.set_page_config("Linear Regression", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        slt.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

slt.markdown("""
    <div class="card">
        <h1>Linear Regression</h1>
        <p> Predict <b> Tip Amount </b> from <b> Total bill </b> using linear Regression.</p>
    </div>
""", unsafe_allow_html=True)

@slt.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Preview
#slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("DataSet Preview")
slt.dataframe(df.head())
#slt.markdown('</div>', unsafe_allow_html=True)

# Data Preparation
x, y = df[['total_bill']], df['tip']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model (FIXED)
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# Plotting
#slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Bills VS Tips")

fig, ax = plt.subplots()
ax.scatter(df['total_bill'], df['tip'], alpha=0.6)
ax.plot(
    df['total_bill'],
    model.predict(scaler.transform(x)),
    color="red"
)
ax.set_xlabel("Total Bills")
ax.set_ylabel("Tips")

slt.pyplot(fig)
#slt.markdown('</div>', unsafe_allow_html=True)

# Metrics - Performance
#slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Model Performances")

c1, c2 = slt.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")

c3, c4 = slt.columns(2)
c3.metric("R2", f"{r2:.2f}")
c4.metric("Adj R2", f"{adj_r2:.2f}")

#slt.markdown('</div>', unsafe_allow_html=True)

# Slope and intercept
slt.markdown(f"""
    <div class="card">
        <h3>Model Interpretation</h3>
        <p><b>Coefficient:</b> {model.coef_[0]:.3f}</p>
        <p><b>Intercept:</b> {model.intercept_:.3f}</p>
    </div>
""", unsafe_allow_html=True)

# Predictions
#slt.markdown('<div class="card">', unsafe_allow_html=True)

bill = slt.slider(
    "Enter the bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

slt.markdown(
    f'<div class="prediction-box"> Predicted tip : $ {tip:.2f}</div>',
    unsafe_allow_html=True
)

#slt.markdown('</div>', unsafe_allow_html=True)
