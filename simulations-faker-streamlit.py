# Install necessary libraries (uncomment the line below to install dependencies)
# !pip install streamlit pandas numpy scikit-learn matplotlib seaborn faker

import streamlit as st
from faker import Faker
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Streamlit App Title
st.title("Synthetic Data Generator and Model Explorer")
st.sidebar.header("Settings")

# Sidebar settings for data generation
num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=1000, value=500, step=50)
random_state = st.sidebar.slider("Random Seed", min_value=0, max_value=100, value=42)

# Step 1: Data Generation
@st.cache
def generate_data(num_samples, random_state):
    fake = Faker()
    np.random.seed(random_state)
    random.seed(random_state)

    data = pd.DataFrame({
        'Feature_1': [fake.random_number(digits=5, fix_len=True) for _ in range(num_samples)],
        'Feature_2': [fake.random_number(digits=3, fix_len=True) for _ in range(num_samples)],
        'Feature_3': [fake.date_between(start_date='-5y', end_date='today').month for _ in range(num_samples)],
        'Feature_4': [random.choice(['A', 'B', 'C', 'D']) for _ in range(num_samples)],
        'Noise': np.random.normal(loc=0, scale=5, size=num_samples),  # Random noise
    })

    # Encode Feature_4
    data['Feature_4_encoded'] = data['Feature_4'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})

    # Stronger relationship between features and target
    data['Target'] = (
        0.5 * data['Feature_1'] +
        1.5 * data['Feature_2'] +
        2 * data['Feature_4_encoded'] -
        1.2 * data['Feature_3'] +
        data['Noise']
    )
    return data

data = generate_data(num_samples, random_state)

# Step 2: Data Exploration
st.subheader("Generated Synthetic Data")
st.write(data.head())

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Pairplot
st.subheader("Feature Pairplot")
sns.pairplot(data.drop(columns=['Feature_4']), hue='Feature_3', palette='viridis')
st.pyplot(plt)

# Step 3: Modeling
st.subheader("Model Training and Evaluation")

# Feature-Target Split
X = data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4_encoded']].values
y = data['Target'].values

# Train-Test Split
test_size = st.sidebar.slider("Test Size (Fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model Selection
model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "Linear Regression", "Polynomial Regression"])

if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=200, value=100, step=10)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

elif model_type == "Linear Regression":
    model = LinearRegression()

elif model_type == "Polynomial Regression":
    degree = st.sidebar.slider("Polynomial Degree", min_value=2, max_value=5, value=2)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)
    model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R-squared:** {r2:.2f}")

# Scatter plot of Predictions vs Actual
st.subheader("True vs Predicted Values")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.set_xlabel("True Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"True vs Predicted ({model_type})")
st.pyplot(fig)

# Save the data and results
if st.button("Save Data and Model Results"):
    data.to_csv("streamlit_synthetic_data.csv", index=False)
    st.success("Data saved as 'streamlit_synthetic_data.csv'!")
