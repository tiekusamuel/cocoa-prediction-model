# cocoa_yield_dashboard.py
# Streamlit dashboard for cocoa yield prediction in Ghana
# -------------------------------------------------------
# This script:
# 1. Loads cocoa_data.csv
# 2. Handles missing values automatically using median imputation
# 3. Trains an XGBoost Regressor
# 4. Tunes hyperparameters using RandomizedSearchCV
# 5. Builds an interactive Streamlit dashboard with:
#    - Sidebar sliders for live input
#    - Predicted yield metric card
#    - Feature importance chart
#    - Rainfall sensitivity chart
#
# To run:
#   streamlit run cocoa_yield_dashboard.py
#
# Required packages:
#   pip install streamlit pandas numpy scikit-learn xgboost plotly

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -------------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Cocoa Yield Prediction Dashboard",
    page_icon="🍫",
    layout="wide"
)

# -------------------------------------------------------
# Simple custom styling for a professional dashboard look
# -------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f8f9fb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2f3b52;
    }
    .stMetric {
        background-color: white;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .info-box {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e6e9ef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Main title and description
# -------------------------------------------------------
st.title("🍫 Cocoa Yield Prediction Dashboard")
st.markdown(
    "Interactive machine learning dashboard for **predicting cocoa yield in Ghana** "
    "based on climate and soil conditions."
)

# -------------------------------------------------------
# Define expected feature columns and target column
# -------------------------------------------------------
FEATURE_COLUMNS = [
    "rainfall",
    "temperature",
    "solar_radiation",
    "soil_ph",
    "phosphorus",
    "nitrogen",
    "potassium"
]
TARGET_COLUMN = "yield_kg_ha"
DATA_FILE = "agriculture_dataset.csv"

# -------------------------------------------------------
# Function to load and validate the dataset
# -------------------------------------------------------
@st.cache_data
def load_data(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)

    # Check that all required columns exist
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"The dataset is missing these required columns: {missing_cols}")

    # Keep only the required columns
    df = df[required_cols].copy()
    return df

# -------------------------------------------------------
# Function to train the model with hyperparameter tuning
# -------------------------------------------------------
@st.cache_resource
def train_model(df):
    # Split features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Build a pipeline:
    # 1. Fill missing values using median
    # 2. Train XGBoost regressor
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Hyperparameter search space for RandomizedSearchCV
    param_dist = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6, 8, 10],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.2, 0.3, 0.5]
    }

    # Randomized search to find good hyperparameters automatically
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="r2",
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    # Train the tuned model
    random_search.fit(X_train, y_train)

    # Best model after tuning
    best_model = random_search.best_estimator_

    # Predictions on test set
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return best_model, metrics, X_train, X_test, y_train, y_test, random_search.best_params_

# -------------------------------------------------------
# Check that the dataset exists
# -------------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.error(
        f"Dataset file '{DATA_FILE}' was not found in the current folder.\n\n"
        f"Please place your CSV file in the same directory as this script."
    )
    st.stop()

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -------------------------------------------------------
# Train model
# -------------------------------------------------------
with st.spinner("Training XGBoost model and tuning hyperparameters..."):
    try:
        model, metrics, X_train, X_test, y_train, y_test, best_params = train_model(df)
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.stop()

# -------------------------------------------------------
# Sidebar inputs for live prediction
# -------------------------------------------------------
st.sidebar.header("🔧 Input Farming Conditions")

# Helper function to create slider ranges from dataset
def slider_bounds(series, fallback_min=0.0, fallback_max=100.0):
    s = pd.to_numeric(series, errors="coerce")
    s_non_null = s.dropna()

    if len(s_non_null) == 0:
        return fallback_min, fallback_max, (fallback_min + fallback_max) / 2

    min_val = float(s_non_null.min())
    max_val = float(s_non_null.max())
    median_val = float(s_non_null.median())

    if min_val == max_val:
        min_val = min_val - 1
        max_val = max_val + 1

    return min_val, max_val, median_val

# Create sliders dynamically for all features
user_inputs = {}
for feature in FEATURE_COLUMNS:
    f_min, f_max, f_default = slider_bounds(df[feature])

    # Set step size automatically based on variable range
    value_range = f_max - f_min
    step = max(value_range / 100, 0.01)

    user_inputs[feature] = st.sidebar.slider(
        label=feature.replace("_", " ").title(),
        min_value=float(f_min),
        max_value=float(f_max),
        value=float(f_default),
        step=float(step)
    )

# Convert user inputs into a DataFrame for prediction
input_df = pd.DataFrame([user_inputs])

# -------------------------------------------------------
# Make live prediction
# -------------------------------------------------------
predicted_yield = model.predict(input_df)[0]

# -------------------------------------------------------
# Top summary section
# -------------------------------------------------------
st.markdown("## Dashboard Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Yield", f"{predicted_yield:,.2f} kg/ha")
col2.metric("Model R² Score", f"{metrics['R2 Score']:.3f}")
col3.metric("Model RMSE", f"{metrics['RMSE']:.2f}")

# -------------------------------------------------------
# Display input summary and model notes
# -------------------------------------------------------
left_info, right_info = st.columns([1.2, 1])

with left_info:
    st.markdown("### Selected Scenario")
    st.dataframe(
        input_df.rename(columns=lambda x: x.replace("_", " ").title()),
        use_container_width=True
    )

with right_info:
    st.markdown("### Model Information")
    st.markdown(
        f"""
        <div class="info-box">
        <b>Algorithm:</b> XGBoost Regressor<br>
        <b>Missing Value Handling:</b> Median Imputation<br>
        <b>Train/Test Split:</b> 80/20<br>
        <b>Hyperparameter Search:</b> RandomizedSearchCV
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------
# Expandable section to show best hyperparameters
# -------------------------------------------------------
with st.expander("See Best Hyperparameters Found"):
    best_params_clean = {k.replace("model__", ""): v for k, v in best_params.items()}
    st.json(best_params_clean)

# -------------------------------------------------------
# Feature importance chart
# -------------------------------------------------------
st.markdown("## Feature Importance")

# Extract trained XGBoost model from pipeline
xgb_model = model.named_steps["model"]

# Get feature importances from XGBoost
feature_importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=True)

# Create horizontal bar chart with Plotly
fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Which Factors Drive Cocoa Yield the Most?",
    color="Importance",
    color_continuous_scale="YlGnBr"
)
fig_importance.update_layout(
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    template="plotly_white",
    height=450
)
st.plotly_chart(fig_importance, use_container_width=True)

# -------------------------------------------------------
# Sensitivity analysis chart for rainfall
# -------------------------------------------------------
st.markdown("## Rainfall Sensitivity Analysis")
st.markdown(
    "This chart shows how predicted yield changes as **rainfall** increases, "
    "while all other conditions remain fixed at your selected sidebar values."
)

# Create rainfall range based on observed dataset values
rain_min = float(df["rainfall"].min())
rain_max = float(df["rainfall"].max())
rainfall_values = np.linspace(rain_min, rain_max, 100)

# Build rows where only rainfall changes and other variables stay fixed
sensitivity_df = pd.DataFrame({
    "rainfall": rainfall_values,
    "temperature": user_inputs["temperature"],
    "solar_radiation": user_inputs["solar_radiation"],
    "soil_ph": user_inputs["soil_ph"],
    "phosphorus": user_inputs["phosphorus"],
    "nitrogen": user_inputs["nitrogen"],
    "potassium": user_inputs["potassium"]
})

# Predict yields for the rainfall sensitivity analysis
sensitivity_predictions = model.predict(sensitivity_df)

# Create interactive line chart
fig_sensitivity = go.Figure()
fig_sensitivity.add_trace(go.Scatter(
    x=rainfall_values,
    y=sensitivity_predictions,
    mode="lines",
    line=dict(width=3, color="#2e8b57"),
    name="Predicted Yield"
))

# Add marker to show current selected rainfall
current_rainfall_pred = model.predict(input_df)[0]
fig_sensitivity.add_trace(go.Scatter(
    x=[user_inputs["rainfall"]],
    y=[current_rainfall_pred],
    mode="markers",
    marker=dict(size=12, color="#8b4513"),
    name="Your Selected Scenario"
))

fig_sensitivity.update_layout(
    title="Yield Response to Increasing Rainfall",
    xaxis_title="Rainfall",
    yaxis_title="Predicted Yield (kg/ha)",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_sensitivity, use_container_width=True)

# -------------------------------------------------------
# Optional data preview section
# -------------------------------------------------------
with st.expander("Preview Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit, XGBoost, Scikit-learn, and Plotly for interactive cocoa yield simulation."
)