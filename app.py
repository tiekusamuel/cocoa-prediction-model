"""
=============================================================================
  🍫 COCOA YIELD PREDICTION DASHBOARD — GHANA
=============================================================================
  Author : Your Name
  Purpose: Interactive web app to predict cocoa yield (kg/ha) based on
           climate and soil conditions using XGBoost + RandomizedSearchCV.
  Run    : streamlit run app.py
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORT ALL REQUIRED LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════
import os
import streamlit as st                          # Web framework for the dashboard
import pandas as pd                             # Data manipulation
import numpy as np                              # Numerical operations
import plotly.express as px                     # Interactive visualizations
import plotly.graph_objects as go               # Advanced Plotly charts
from sklearn.model_selection import (           # ML utilities
    train_test_split,
    RandomizedSearchCV
)
from sklearn.metrics import (                   # Model evaluation metrics
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor               # The XGBoost model
from scipy.stats import uniform, randint        # Distributions for hyperparameter search
import warnings
warnings.filterwarnings('ignore')               # Suppress sklearn warnings for cleaner output

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: PAGE CONFIGURATION & CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════

# Configure the Streamlit page (must be the FIRST Streamlit command)
st.set_page_config(
    page_title="🍫 Ghana Cocoa Yield Predictor",
    page_icon="🍫",
    layout="wide",                              # Use the full width of the browser
    initial_sidebar_state="expanded"            # Sidebar open by default
)

# Inject custom CSS for professional styling
st.markdown("""
<style>
    /* ── Main background and text ── */
    .main {
        background-color: #faf8f5;
    }

    /* ── Metric card styling ── */
    .metric-card {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c23 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin-bottom: 10px;
    }
    .metric-card h2 {
        font-size: 2.5rem;
        margin: 0;
        color: #f5e6a3;
        font-weight: 800;
    }
    .metric-card p {
        font-size: 1rem;
        margin: 5px 0 0 0;
        color: #d4e8c2;
    }

    /* ── Performance metric cards (smaller) ── */
    .perf-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        border-left: 5px solid #4a7c23;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .perf-card h3 {
        font-size: 1.6rem;
        margin: 0;
        color: #2d5016;
    }
    .perf-card p {
        font-size: 0.85rem;
        margin: 5px 0 0 0;
        color: #666;
    }

    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(90deg, #2d5016, #4a7c23);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* ── Info box ── */
    .info-box {
        background: #e8f5e9;
        border-left: 5px solid #4a7c23;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #333;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a0a 0%, #2d5016 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div {
        color: white;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA LOADING WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data  # Cache the data so it only loads once (speeds up reruns)
def load_data():
    """
    Load the cocoa dataset from CSV.
    Returns the DataFrame or None if the file is not found.
    """
    try:
        df = pd.read_csv('cocoa_data.csv')
        return df
    except FileNotFoundError:
        return None

# Load the data
data = load_data()

# If the file doesn't exist, show an error and stop
if data is None:
    st.error("🚨 **File Not Found:** `cocoa_data.csv` is missing!")
    st.info("""
    **How to fix this:**
    1. Save the `generate_data.py` script from the instructions
    2. Run `python generate_data.py` in the same folder as this app
    3. Then run `streamlit run app.py` again
    """)
    st.stop()  # Halt execution here

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA PREPROCESSING — HANDLE MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_data(df):
    """
    Clean the dataset:
    1. Fill missing values with the median of each column
    2. Separate features (X) from the target variable (y)

    Using median (not mean) because median is robust to outliers.
    """
    # Count missing values before filling
    missing_before = df.isnull().sum().sum()

    # Fill every column's missing values with that column's median
    df_clean = df.fillna(df.median())

    # Define feature columns (everything except the target)
    feature_cols = ['rainfall', 'temperature', 'solar_radiation',
                    'soil_ph', 'phosphorus', 'nitrogen', 'potassium']

    # Separate into features (X) and target (y)
    X = df_clean[feature_cols]
    y = df_clean['yield_kg_ha']

    return X, y, feature_cols, missing_before

# Run preprocessing
X, y, feature_cols, missing_count = preprocess_data(data)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL BUILDING WITH HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource  # Cache the trained model (persists across reruns)
def build_model(X, y):
    """
    Build and tune an XGBoost Regressor:
    1. Split data 80% train / 20% test
    2. Define a search space of hyperparameters
    3. Use RandomizedSearchCV to find the best combination
    4. Return the trained model and performance metrics
    """
    # ── Step 1: Split the data ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,       # 20% for testing
        random_state=42       # Reproducible split
    )

    # ── Step 2: Define hyperparameter search space ──
    # These are the "knobs" XGBoost can adjust to learn better
    param_distributions = {
        'n_estimators': randint(100, 500),       # Number of boosting rounds (trees)
        'max_depth': randint(3, 10),             # How deep each tree can grow
        'learning_rate': uniform(0.01, 0.29),    # Step size for each round (0.01 to 0.30)
        'subsample': uniform(0.6, 0.4),          # Fraction of samples per tree (0.6 to 1.0)
        'colsample_bytree': uniform(0.6, 0.4),   # Fraction of features per tree
        'min_child_weight': randint(1, 10),      # Minimum samples in a leaf
        'reg_alpha': uniform(0, 1),              # L1 regularization (prevents overfitting)
        'reg_lambda': uniform(0, 2),             # L2 regularization
    }

    # ── Step 3: Create base XGBoost model ──
    xgb_base = XGBRegressor(
        objective='reg:squarederror',  # Regression task
        random_state=42,
        verbosity=0                    # Suppress training logs
    )

    # ── Step 4: Run RandomizedSearchCV ──
    # This tries 50 random combinations and picks the best one
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=50,                   # Try 50 random combinations
        scoring='r2',                # Optimize for R² score
        cv=3,                        # 3-fold cross-validation
        random_state=42,
        n_jobs=-1,                   # Use all CPU cores
        verbose=0
    )

    # Fit the search on training data
    search.fit(X_train, y_train)

    # ── Step 5: Get the best model and evaluate ──
    best_model = search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate performance metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),                       # R² (1.0 = perfect)
        'mae': mean_absolute_error(y_test, y_pred),            # Mean Absolute Error
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),   # Root Mean Squared Error
        'best_params': search.best_params_,                     # The winning parameters
        'y_test': y_test,                                       # Actual values (for plotting)
        'y_pred': y_pred,                                       # Predicted values
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    return best_model, metrics

# Build the model (cached — only runs once unless data changes)
with st.spinner('🔧 Training XGBoost model with hyperparameter tuning... Please wait.'):
    model, metrics = build_model(X, y)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: SIDEBAR — INTERACTIVE INPUT SLIDERS
# ═══════════════════════════════════════════════════════════════════════════

# Sidebar header
st.sidebar.markdown("## 🍫 Cocoa Yield Predictor")
st.sidebar.markdown("### 🇬🇭 Ghana Farming Simulator")
st.sidebar.markdown("---")
st.sidebar.markdown("**Adjust the conditions below to simulate different farming scenarios:**")
st.sidebar.markdown("")

# ── Climate Variables ──
st.sidebar.markdown("#### 🌦️ Climate Conditions")

# Rainfall slider (mm/year) — Ghana cocoa belt typically gets 1000-1800mm
rainfall_input = st.sidebar.slider(
    "🌧️ Annual Rainfall (mm)",
    min_value=600,
    max_value=2200,
    value=1400,                    # Default: typical for Western Region
    step=25,
    help="Ghana cocoa belt: 1,000–1,800 mm/year. Western Region ≈1,400 mm."
)

# Temperature slider (°C) — Cocoa grows best between 24-28°C
temperature_input = st.sidebar.slider(
    "🌡️ Avg Temperature (°C)",
    min_value=20.0,
    max_value=34.0,
    value=26.5,
    step=0.5,
    help="Optimal range: 24–28°C. Above 30°C causes heat stress."
)

# Solar radiation slider (MJ/m²/day)
solar_input = st.sidebar.slider(
    "☀️ Solar Radiation (MJ/m²/day)",
    min_value=10.0,
    max_value=28.0,
    value=18.0,
    step=0.5,
    help="Cocoa needs moderate light. Full sun: ~20+ MJ/m²/day."
)

st.sidebar.markdown("")

# ── Soil Variables ──
st.sidebar.markdown("#### 🌱 Soil Conditions")

# Soil pH — Cocoa prefers slightly acidic (5.5-6.5)
ph_input = st.sidebar.slider(
    "🧪 Soil pH",
    min_value=4.0,
    max_value=8.0,
    value=6.0,
    step=0.1,
    help="Optimal: 5.5–6.5. Below 5.0 = too acidic, above 7.0 = too alkaline."
)

# Phosphorus (mg/kg)
phosphorus_input = st.sidebar.slider(
    "🟠 Phosphorus (mg/kg)",
    min_value=1.0,
    max_value=30.0,
    value=12.0,
    step=0.5,
    help="Bray-1 extractable P. Above 10 mg/kg is adequate for cocoa."
)

# Nitrogen (% total N)
nitrogen_input = st.sidebar.slider(
    "🟢 Nitrogen (%)",
    min_value=0.05,
    max_value=0.40,
    value=0.18,
    step=0.01,
    help="Total soil nitrogen. Above 0.15% is generally adequate."
)

# Potassium (cmol/kg)
potassium_input = st.sidebar.slider(
    "🟡 Potassium (cmol/kg)",
    min_value=0.05,
    max_value=0.80,
    value=0.35,
    step=0.01,
    help="Exchangeable K. Above 0.25 cmol/kg is adequate for cocoa."
)

# ── Make a live prediction using the slider values ──
# Create a DataFrame with exactly the same column names as training data
input_data = pd.DataFrame({
    'rainfall': [rainfall_input],
    'temperature': [temperature_input],
    'solar_radiation': [solar_input],
    'soil_ph': [ph_input],
    'phosphorus': [phosphorus_input],
    'nitrogen': [nitrogen_input],
    'potassium': [potassium_input]
})

# Get the model's prediction for these conditions
predicted_yield = model.predict(input_data)[0]
##
##Appending new input data and predicted yield to the original dataset for potential future use (e.g., retraining or analysis) 

row = pd.DataFrame({
    'rainfall': [rainfall_input],
    'temperature': [temperature_input],
    'solar_radiation': [solar_input],
    'soil_ph': [ph_input],
    'phosphorus': [phosphorus_input],
    'nitrogen': [nitrogen_input],
    'potassium': [potassium_input],
    'yield_kg_ha': [predicted_yield]
})
file_path = "cocoa_data.csv"

if not os.path.exists(file_path):
    row.to_csv(file_path, mode='w', header=True, index=False)
else:
    row.to_csv(file_path, mode='a', header=False, index=False)


# Show the prediction prominently in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Live Prediction")
st.sidebar.markdown(
    f"""
    <div style='background: linear-gradient(135deg, #f5e6a3, #e8c84a);
                padding: 20px; border-radius: 12px; text-align: center;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);'>
        <h1 style='color: #2d5016; margin: 0; font-size: 2.2rem;'>
            {predicted_yield:,.0f} kg/ha
        </h1>
        <p style='color: #4a6b2a; margin: 5px 0 0 0; font-size: 0.95rem;'>
            Predicted Cocoa Yield
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Classify the yield level for user guidance
if predicted_yield >= 800:
    yield_level = "🟢 Excellent"
    yield_advice = "These conditions are very favorable for cocoa production!"
elif predicted_yield >= 500:
    yield_level = "🟡 Good"
    yield_advice = "Decent conditions. Consider optimizing soil nutrients."
elif predicted_yield >= 350:
    yield_level = "🟠 Below Average"
    yield_advice = "Room for improvement. Check soil pH and nutrient levels."
else:
    yield_level = "🔴 Poor"
    yield_advice = "Conditions are challenging. Major interventions needed."

st.sidebar.markdown(f"**Status:** {yield_level}")
st.sidebar.info(yield_advice)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN PAGE — HEADER AND OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

# Page title
st.markdown("""
    <div style='text-align: center; padding: 10px 0 5px 0;'>
        <h1 style='color: #2d5016; font-size: 2.5rem; margin-bottom: 0;'>
            🍫 Ghana Cocoa Yield Prediction Dashboard
        </h1>
        <p style='color: #666; font-size: 1.1rem; margin-top: 5px;'>
            Machine Learning-Powered Agricultural Decision Support System
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Row of metric cards: Prediction + Model Performance ──
col1, col2, col3, col4 = st.columns(4)

# Card 1: The main prediction
with col1:
    st.markdown(f"""
        <div class='metric-card'>
            <h2>{predicted_yield:,.0f}</h2>
            <p>🎯 Predicted Yield (kg/ha)</p>
        </div>
    """, unsafe_allow_html=True)

# Card 2: R² Score (how well the model fits — 1.0 is perfect)
with col2:
    st.markdown(f"""
        <div class='perf-card'>
            <h3>{metrics['r2']:.3f}</h3>
            <p>📊 R² Score (Model Accuracy)</p>
        </div>
    """, unsafe_allow_html=True)

# Card 3: Mean Absolute Error (average prediction error in kg/ha)
with col3:
    st.markdown(f"""
        <div class='perf-card'>
            <h3>{metrics['mae']:.1f}</h3>
            <p>📏 MAE (kg/ha Error)</p>
        </div>
    """, unsafe_allow_html=True)

# Card 4: Dataset info
with col4:
    st.markdown(f"""
        <div class='perf-card'>
            <h3>{len(data):,}</h3>
            <p>📋 Training Observations</p>
        </div>
    """, unsafe_allow_html=True)

# ── Info box about the data ──
st.markdown(f"""
<div class='info-box'>
    <strong>📌 Data Summary:</strong> The model was trained on <strong>{metrics['train_size']}</strong>
    observations and tested on <strong>{metrics['test_size']}</strong> holdout samples.
    <strong>{missing_count}</strong> missing values were automatically filled using median imputation.
    Hyperparameter tuning tested <strong>50</strong> configurations via 3-fold cross-validation.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: FEATURE IMPORTANCE CHART
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Feature Importance — What Drives Cocoa Yield?</div>',
            unsafe_allow_html=True)

# Get feature importances from the trained XGBoost model
# XGBoost assigns an importance score to each feature based on how much
# it contributed to reducing prediction error across all trees
importances = model.feature_importances_

# Create a DataFrame for plotting, sorted by importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)  # Sort ascending for horizontal bar

# Create readable labels for the features
label_map = {
    'rainfall': '🌧️ Rainfall',
    'temperature': '🌡️ Temperature',
    'solar_radiation': '☀️ Solar Radiation',
    'soil_ph': '🧪 Soil pH',
    'phosphorus': '🟠 Phosphorus',
    'nitrogen': '🟢 Nitrogen',
    'potassium': '🟡 Potassium'
}
importance_df['Label'] = importance_df['Feature'].map(label_map)

# Build the horizontal bar chart using Plotly
fig_importance = go.Figure()

fig_importance.add_trace(go.Bar(
    x=importance_df['Importance'],
    y=importance_df['Label'],
    orientation='h',                            # Horizontal bars
    marker=dict(
        color=importance_df['Importance'],       # Color by importance value
        colorscale='YlGn',                       # Yellow-Green color scale (cocoa theme!)
        line=dict(color='#2d5016', width=1.5)
    ),
    text=[f'{v:.1%}' for v in importance_df['Importance']],   # Show percentage labels
    textposition='outside',
    textfont=dict(size=13, color='#333')
))

fig_importance.update_layout(
    title=dict(
        text='Which Factors Have the Biggest Impact on Cocoa Yield?',
        font=dict(size=16, color='#2d5016'),
        x=0.5                                    # Center the title
    ),
    xaxis_title='Relative Importance Score',
    yaxis_title='',
    height=420,
    plot_bgcolor='rgba(0,0,0,0)',                # Transparent background
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=80, t=50, b=40),
    xaxis=dict(showgrid=True, gridcolor='#eee', tickformat='.0%'),
    yaxis=dict(showgrid=False)
)

# Display the chart
st.plotly_chart(fig_importance, use_container_width=True)

# Explanation text
top_feature = importance_df.iloc[-1]['Label']     # Most important feature
st.markdown(f"""
<div class='info-box'>
    <strong>💡 Interpretation:</strong> {top_feature} is the most important predictor of cocoa yield
    in this model. Features with higher importance scores have more influence on whether yield
    will be high or low. This can help prioritize which farming conditions to focus on.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: SENSITIVITY ANALYSIS — HOW YIELD CHANGES WITH RAINFALL
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📈 Sensitivity Analysis — Yield vs. Rainfall</div>',
            unsafe_allow_html=True)

st.markdown("""
This chart shows how predicted yield changes as **rainfall increases from 600mm to 2200mm**,
while keeping **all other variables fixed** at your current slider values.
This helps you understand the marginal effect of rainfall on yield.
""")

# Create a range of rainfall values to test
rainfall_range = np.linspace(600, 2200, 100)    # 100 points from 600 to 2200 mm

# For each rainfall value, predict yield with the user's other inputs held constant
sensitivity_predictions = []
for rain in rainfall_range:
    scenario = pd.DataFrame({
        'rainfall': [rain],
        'temperature': [temperature_input],       # User's selected temperature
        'solar_radiation': [solar_input],          # User's selected solar radiation
        'soil_ph': [ph_input],                     # User's selected soil pH
        'phosphorus': [phosphorus_input],           # User's selected phosphorus
        'nitrogen': [nitrogen_input],               # User's selected nitrogen
        'potassium': [potassium_input]              # User's selected potassium
    })
    pred = model.predict(scenario)[0]
    sensitivity_predictions.append(pred)

# Build the sensitivity DataFrame
sensitivity_df = pd.DataFrame({
    'Rainfall (mm)': rainfall_range,
    'Predicted Yield (kg/ha)': sensitivity_predictions
})

# Create the line chart
fig_sensitivity = go.Figure()

# Main line
fig_sensitivity.add_trace(go.Scatter(
    x=sensitivity_df['Rainfall (mm)'],
    y=sensitivity_df['Predicted Yield (kg/ha)'],
    mode='lines',
    name='Predicted Yield',
    line=dict(color='#4a7c23', width=3.5),
    fill='tozeroy',                              # Fill area under the curve
    fillcolor='rgba(74, 124, 35, 0.1)',
))

# Add a vertical marker showing the user's current rainfall selection
fig_sensitivity.add_vline(
    x=rainfall_input,
    line_dash="dash",
    line_color="#e8c84a",
    line_width=2,
    annotation_text=f"Your Selection: {rainfall_input}mm",
    annotation_position="top",
    annotation_font_size=12,
    annotation_font_color="#8b6914"
)

# Add a point marker at the user's current prediction
fig_sensitivity.add_trace(go.Scatter(
    x=[rainfall_input],
    y=[predicted_yield],
    mode='markers+text',
    name='Current Prediction',
    marker=dict(size=14, color='#e8c84a', line=dict(width=2, color='#2d5016')),
    text=[f'{predicted_yield:,.0f} kg/ha'],
    textposition='top center',
    textfont=dict(size=13, color='#2d5016', family='Arial Black')
))

fig_sensitivity.update_layout(
    title=dict(
        text='How Does Changing Rainfall Affect Yield? (Other Conditions Fixed)',
        font=dict(size=16, color='#2d5016'),
        x=0.5
    ),
    xaxis_title='Annual Rainfall (mm)',
    yaxis_title='Predicted Yield (kg/ha)',
    height=450,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='#eee'),
    yaxis=dict(showgrid=True, gridcolor='#eee'),
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

# Display the chart
st.plotly_chart(fig_sensitivity, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: ACTUAL vs PREDICTED SCATTER PLOT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🎯 Model Validation — Actual vs. Predicted Yield</div>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    # Create a scatter plot comparing actual test values to predictions
    validation_df = pd.DataFrame({
        'Actual Yield (kg/ha)': metrics['y_test'].values,
        'Predicted Yield (kg/ha)': metrics['y_pred']
    })

    fig_scatter = px.scatter(
        validation_df,
        x='Actual Yield (kg/ha)',
        y='Predicted Yield (kg/ha)',
        opacity=0.5,
        color_discrete_sequence=['#4a7c23']
    )

    # Add the perfect prediction line (45-degree line)
    min_val = min(validation_df['Actual Yield (kg/ha)'].min(),
                  validation_df['Predicted Yield (kg/ha)'].min())
    max_val = max(validation_df['Actual Yield (kg/ha)'].max(),
                  validation_df['Predicted Yield (kg/ha)'].max())

    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig_scatter.update_layout(
        title=dict(text='How Close Are Predictions to Reality?',
                   font=dict(size=15, color='#2d5016'), x=0.5),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee'),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    # Show the best hyperparameters found by RandomizedSearchCV
    st.markdown("#### ⚙️ Best Hyperparameters")
    st.markdown("Found via RandomizedSearchCV (50 iterations, 3-fold CV):")

    # Display each parameter in a clean format
    params = metrics['best_params']
    param_display = {
        'Number of Trees': params.get('n_estimators', 'N/A'),
        'Max Tree Depth': params.get('max_depth', 'N/A'),
        'Learning Rate': f"{params.get('learning_rate', 0):.4f}",
        'Subsample Ratio': f"{params.get('subsample', 0):.3f}",
        'Column Sample/Tree': f"{params.get('colsample_bytree', 0):.3f}",
        'Min Child Weight': params.get('min_child_weight', 'N/A'),
        'L1 Regularization': f"{params.get('reg_alpha', 0):.4f}",
        'L2 Regularization': f"{params.get('reg_lambda', 0):.4f}",
    }

    for param_name, param_value in param_display.items():
        st.markdown(f"- **{param_name}:** `{param_value}`")

    st.markdown(f"""
    <div class='info-box'>
        <strong>📐 Model Performance:</strong><br>
        • R² = {metrics['r2']:.4f} (explains {metrics['r2']*100:.1f}% of yield variance)<br>
        • MAE = {metrics['mae']:.1f} kg/ha (avg error)<br>
        • RMSE = {metrics['rmse']:.1f} kg/ha (penalizes large errors)
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: MULTI-VARIABLE SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🔬 Multi-Variable Sensitivity Explorer</div>',
            unsafe_allow_html=True)

st.markdown("""
Choose **any variable** to see how yield responds to changes in that factor,
while keeping all other inputs at your current slider values.
""")

# Let the user pick which variable to analyze
variable_labels = {
    'rainfall': '🌧️ Rainfall (mm)',
    'temperature': '🌡️ Temperature (°C)',
    'solar_radiation': '☀️ Solar Radiation (MJ/m²/day)',
    'soil_ph': '🧪 Soil pH',
    'phosphorus': '🟠 Phosphorus (mg/kg)',
    'nitrogen': '🟢 Nitrogen (%)',
    'potassium': '🟡 Potassium (cmol/kg)'
}

# Define the range for each variable
variable_ranges = {
    'rainfall': (600, 2200),
    'temperature': (20, 34),
    'solar_radiation': (10, 28),
    'soil_ph': (4.0, 8.0),
    'phosphorus': (1, 30),
    'nitrogen': (0.05, 0.40),
    'potassium': (0.05, 0.80)
}

# Current values from sliders
current_values = {
    'rainfall': rainfall_input,
    'temperature': temperature_input,
    'solar_radiation': solar_input,
    'soil_ph': ph_input,
    'phosphorus': phosphorus_input,
    'nitrogen': nitrogen_input,
    'potassium': potassium_input
}

# User selects the variable
selected_var = st.selectbox(
    "Select variable to analyze:",
    options=list(variable_labels.keys()),
    format_func=lambda x: variable_labels[x],
    index=0
)

# Generate predictions across the range of the selected variable
var_range = np.linspace(variable_ranges[selected_var][0],
                        variable_ranges[selected_var][1], 100)

multi_predictions = []
for val in var_range:
    scenario = current_values.copy()   # Start with user's current selections
    scenario[selected_var] = val       # Override just the selected variable
    scenario_df = pd.DataFrame({k: [v] for k, v in scenario.items()})
    pred = model.predict(scenario_df)[0]
    multi_predictions.append(pred)

# Create the chart
fig_multi = go.Figure()

fig_multi.add_trace(go.Scatter(
    x=var_range,
    y=multi_predictions,
    mode='lines',
    line=dict(color='#4a7c23', width=3),
    fill='tozeroy',
    fillcolor='rgba(74, 124, 35, 0.08)',
    name='Predicted Yield'
))

# Mark current selection
fig_multi.add_vline(
    x=current_values[selected_var],
    line_dash="dash", line_color="#e8c84a", line_width=2,
    annotation_text=f"Current: {current_values[selected_var]}",
    annotation_position="top"
)

fig_multi.update_layout(
    title=dict(
        text=f'Yield Response to {variable_labels[selected_var]}',
        font=dict(size=16, color='#2d5016'), x=0.5
    ),
    xaxis_title=variable_labels[selected_var],
    yaxis_title='Predicted Yield (kg/ha)',
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='#eee'),
    yaxis=dict(showgrid=True, gridcolor='#eee'),
)

st.plotly_chart(fig_multi, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📋 Raw Data Explorer</div>',
            unsafe_allow_html=True)

# Expandable section to view the raw data
with st.expander("🔍 Click to view the dataset", expanded=False):
    st.dataframe(
        data.style.format(precision=2).highlight_null(color='#ffcccc'),
        use_container_width=True,
        height=350
    )

    # Show basic statistics
    st.markdown("#### 📊 Summary Statistics")
    st.dataframe(
        data.describe().style.format(precision=3),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px 0;'>
    <p style='font-size: 0.9rem;'>
        🍫 <strong>Ghana Cocoa Yield Prediction Dashboard</strong> |
        Built with Streamlit, XGBoost, Plotly & Scikit-learn<br>
        <em>For research and educational purposes. Model predictions are simulations,
        not guarantees.</em>
    </p>
</div>
""", unsafe_allow_html=True)