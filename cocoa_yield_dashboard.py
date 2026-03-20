"""
Cocoa Yield Prediction Dashboard for Ghana
============================================
Interactive web application for predicting cocoa yields based on climate and soil conditions.
Built with Streamlit, XGBoost, and Plotly for professional data visualization.

Author: Data Science Pipeline
Purpose: Help farmers and researchers simulate different farming scenarios
"""

# ============================================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 2: PAGE CONFIGURATION AND STYLING
# ============================================================================

# Configure the Streamlit page with custom settings
st.set_page_config(
    page_title="Cocoa Yield Predictor - Ghana",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2 {
        color: #34495e;
        border-bottom: 3px solid #e67e22;
        padding-bottom: 10px;
    }
    h3 {
        color: #7f8c8d;
    }
    .stAlert {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING FUNCTION
# ============================================================================

@st.cache_data
def load_and_prepare_data(filepath):
    """
    Load the cocoa dataset and handle missing values automatically.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing cocoa data
        
    Returns:
    --------
    df : pandas.DataFrame
        Cleaned dataset with missing values filled
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Fill missing values with median for each column
        # Median is robust to outliers, better than mean for agricultural data
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        
        return df
    
    except FileNotFoundError:
        st.error(f"❌ Error: Could not find the file '{filepath}'. Please ensure cocoa_data.csv is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# ============================================================================
# SECTION 4: MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================

@st.cache_resource
def train_xgboost_model(df):
    """
    Train an XGBoost model with automatic hyperparameter tuning.
    
    This function:
    1. Splits data into features (X) and target (y)
    2. Creates 80/20 train/test split
    3. Uses RandomizedSearchCV to find best hyperparameters
    4. Trains final model and calculates performance metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The preprocessed cocoa dataset
        
    Returns:
    --------
    model : XGBRegressor
        Trained XGBoost model
    X_train, X_test, y_train, y_test : arrays
        Split datasets for evaluation
    best_params : dict
        Best hyperparameters found
    metrics : dict
        Performance metrics (RMSE, R², MAE)
    """
    
    # Define features (climate and soil variables) and target (yield)
    feature_columns = ['rainfall', 'temp_min', 'temp_max', 'solar_radiation', 
                      'soil_ph', 'phosphorus', 'nitrogen']
    target_column = 'yield_kg_ha'
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Split into training (80%) and testing (20%) sets
    # random_state=42 ensures reproducible results
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define hyperparameter search space
    # These are the parameters we'll automatically tune
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],          # Number of trees
        'max_depth': [3, 5, 7, 9],                      # Maximum tree depth
        'learning_rate': [0.01, 0.05, 0.1, 0.2],       # Step size
        'subsample': [0.6, 0.8, 1.0],                  # Fraction of samples per tree
        'colsample_bytree': [0.6, 0.8, 1.0],           # Fraction of features per tree
        'min_child_weight': [1, 3, 5],                  # Minimum sum of weights in a child
        'gamma': [0, 0.1, 0.2]                          # Minimum loss reduction
    }
    
    # Create base XGBoost model
    xgb_base = XGBRegressor(random_state=42, objective='reg:squarederror')
    
    # Set up RandomizedSearchCV to find best hyperparameters
    # n_iter=20 means we'll try 20 random combinations
    # cv=3 means 3-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Train the model with hyperparameter search
    with st.spinner('🔍 Training XGBoost model with hyperparameter tuning... This may take a minute.'):
        random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    }
    
    return best_model, X_train, X_test, y_train, y_test, random_search.best_params_, metrics

# ============================================================================
# SECTION 5: MAIN APPLICATION INTERFACE
# ============================================================================

def main():
    """
    Main function that builds the Streamlit dashboard interface.
    """
    
    # Header section with title and description
    st.title("🍫 Cocoa Yield Prediction Dashboard")
    st.markdown("### Simulate Farming Scenarios in Ghana")
    st.markdown("---")
    
    # Information alert for users
    st.info("""
    👋 **Welcome!** This dashboard helps you predict cocoa yields based on climate and soil conditions.
    
    📊 **How to use:**
    1. Adjust the sliders in the sidebar to simulate different farming conditions
    2. See your predicted yield instantly
    3. Explore which factors matter most for cocoa production
    """)
    
    # Load the data
    df = load_and_prepare_data('cocoa_data.csv')
    
    # Display dataset statistics in an expander (collapsible section)
    with st.expander("📁 View Dataset Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Average Yield", f"{df['yield_kg_ha'].mean():.2f} kg/ha")
        
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(10), use_container_width=True)
    
    # Train the model
    model, X_train, X_test, y_train, y_test, best_params, metrics = train_xgboost_model(df)
    
    # Display model performance in an expander
    with st.expander("🎯 Model Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['R²']:.4f}", 
                     help="How well the model fits the data (1.0 is perfect)")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f} kg/ha",
                     help="Average prediction error")
        with col3:
            st.metric("MAE", f"{metrics['MAE']:.2f} kg/ha",
                     help="Mean absolute error")
        
        st.markdown("**Best Hyperparameters Found:**")
        st.json(best_params)
    
    # ============================================================================
    # SECTION 6: SIDEBAR - INTERACTIVE INPUT SLIDERS
    # ============================================================================
    
    st.sidebar.header("🌾 Farming Scenario Inputs")
    st.sidebar.markdown("Adjust the parameters below to simulate different conditions:")
    
    # Calculate min, max, and default values from the dataset
    # This ensures sliders have realistic ranges
    
    st.sidebar.subheader("☁️ Climate Variables")
    rainfall = st.sidebar.slider(
        "Rainfall (mm)",
        min_value=float(df['rainfall'].min()),
        max_value=float(df['rainfall'].max()),
        value=float(df['rainfall'].median()),
        step=10.0,
        help="Annual rainfall in millimeters"
    )
    
    temp_min = st.sidebar.slider(
        "Minimum Temperature (°C)",
        min_value=float(df['temp_min'].min()),
        max_value=float(df['temp_min'].max()),
        value=float(df['temp_min'].median()),
        step=0.5,
        help="Average minimum temperature"
    )
    
    temp_max = st.sidebar.slider(
        "Maximum Temperature (°C)",
        min_value=float(df['temp_max'].min()),
        max_value=float(df['temp_max'].max()),
        value=float(df['temp_max'].median()),
        step=0.5,
        help="Average maximum temperature"
    )
    
    solar_radiation = st.sidebar.slider(
        "Solar Radiation (MJ/m²)",
        min_value=float(df['solar_radiation'].min()),
        max_value=float(df['solar_radiation'].max()),
        value=float(df['solar_radiation'].median()),
        step=0.5,
        help="Solar radiation received"
    )
    
    st.sidebar.subheader("🌱 Soil Variables")
    soil_ph = st.sidebar.slider(
        "Soil pH",
        min_value=float(df['soil_ph'].min()),
        max_value=float(df['soil_ph'].max()),
        value=float(df['soil_ph'].median()),
        step=0.1,
        help="Soil acidity/alkalinity level"
    )
    
    phosphorus = st.sidebar.slider(
        "Phosphorus (ppm)",
        min_value=float(df['phosphorus'].min()),
        max_value=float(df['phosphorus'].max()),
        value=float(df['phosphorus'].median()),
        step=1.0,
        help="Phosphorus content in soil"
    )
    
    nitrogen = st.sidebar.slider(
        "Nitrogen (ppm)",
        min_value=float(df['nitrogen'].min()),
        max_value=float(df['nitrogen'].max()),
        value=float(df['nitrogen'].median()),
        step=1.0,
        help="Nitrogen content in soil"
    )
    
    # ============================================================================
    # SECTION 7: LIVE PREDICTION
    # ============================================================================
    
    # Create a DataFrame from user inputs (same format as training data)
    user_input = pd.DataFrame({
        'rainfall': [rainfall],
        'temp_min': [temp_min],
        'temp_max': [temp_max],
        'solar_radiation': [solar_radiation],
        'soil_ph': [soil_ph],
        'phosphorus': [phosphorus],
        'nitrogen': [nitrogen]
    })
    
    # Make prediction using the trained model
    predicted_yield = model.predict(user_input)[0]
    
    # Display prediction prominently
    st.markdown("---")
    st.subheader("🎯 Live Yield Prediction")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(
            label="Predicted Cocoa Yield",
            value=f"{predicted_yield:.2f} kg/ha",
            delta=f"{predicted_yield - df['yield_kg_ha'].mean():.2f} vs. average",
            help="Predicted yield based on your input parameters"
        )
    
    with col2:
        st.metric(
            label="Dataset Average",
            value=f"{df['yield_kg_ha'].mean():.2f} kg/ha",
            help="Average yield in the dataset"
        )
    
    with col3:
        # Calculate percentile ranking
        percentile = (df['yield_kg_ha'] <= predicted_yield).mean() * 100
        st.metric(
            label="Percentile Rank",
            value=f"{percentile:.1f}%",
            help="What % of farms your prediction beats"
        )
    
    # ============================================================================
    # SECTION 8: VISUALIZATION 1 - FEATURE IMPORTANCE
    # ============================================================================
    
    st.markdown("---")
    st.subheader("📊 Feature Importance Analysis")
    st.markdown("Which factors have the biggest impact on cocoa yield?")
    
    # Get feature importances from the trained model
    feature_names = ['Rainfall', 'Min Temp', 'Max Temp', 'Solar Radiation', 
                    'Soil pH', 'Phosphorus', 'Nitrogen']
    importances = model.feature_importances_
    
    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)  # Sort for better visualization
    
    # Create horizontal bar chart using Plotly
    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=importance_df['Importance'].round(3),
        textposition='auto',
    ))
    
    fig_importance.update_layout(
        title="Feature Importance in Cocoa Yield Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        template="plotly_white",
        font=dict(size=12),
        showlegend=False
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Add interpretation
    most_important = importance_df.iloc[-1]
    st.info(f"💡 **Insight:** {most_important['Feature']} is the most important factor, with an importance score of {most_important['Importance']:.3f}")
    
    # ============================================================================
    # SECTION 9: VISUALIZATION 2 - SENSITIVITY ANALYSIS
    # ============================================================================
    
    st.markdown("---")
    st.subheader("📈 Rainfall Sensitivity Analysis")
    st.markdown("How does yield change as rainfall varies? (Other factors held constant at your slider values)")
    
    # Create a range of rainfall values from min to max
    rainfall_range = np.linspace(
        df['rainfall'].min(),
        df['rainfall'].max(),
        50  # 50 points for smooth curve
    )
    
    # Keep all other variables constant at user's selected values
    sensitivity_predictions = []
    
    for rain_val in rainfall_range:
        sensitivity_input = pd.DataFrame({
            'rainfall': [rain_val],
            'temp_min': [temp_min],
            'temp_max': [temp_max],
            'solar_radiation': [solar_radiation],
            'soil_ph': [soil_ph],
            'phosphorus': [phosphorus],
            'nitrogen': [nitrogen]
        })
        pred = model.predict(sensitivity_input)[0]
        sensitivity_predictions.append(pred)
    
    # Create line chart using Plotly
    fig_sensitivity = go.Figure()
    
    # Add the sensitivity curve
    fig_sensitivity.add_trace(go.Scatter(
        x=rainfall_range,
        y=sensitivity_predictions,
        mode='lines',
        name='Predicted Yield',
        line=dict(color='#e67e22', width=3),
        fill='tonexty',
        fillcolor='rgba(230, 126, 34, 0.1)'
    ))
    
    # Add a marker for the user's current selection
    fig_sensitivity.add_trace(go.Scatter(
        x=[rainfall],
        y=[predicted_yield],
        mode='markers',
        name='Your Selection',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig_sensitivity.update_layout(
        title="Yield Sensitivity to Rainfall Changes",
        xaxis_title="Rainfall (mm)",
        yaxis_title="Predicted Yield (kg/ha)",
        height=500,
        template="plotly_white",
        font=dict(size=12),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    # Calculate and display optimal rainfall
    optimal_idx = np.argmax(sensitivity_predictions)
    optimal_rainfall = rainfall_range[optimal_idx]
    optimal_yield = sensitivity_predictions[optimal_idx]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🌧️ **Optimal Rainfall:** {optimal_rainfall:.1f} mm → {optimal_yield:.2f} kg/ha")
    with col2:
        if rainfall < optimal_rainfall:
            st.warning(f"💧 Increasing rainfall by {optimal_rainfall - rainfall:.1f} mm could improve yield by {optimal_yield - predicted_yield:.2f} kg/ha")
        elif rainfall > optimal_rainfall:
            st.warning(f"💧 Reducing rainfall by {rainfall - optimal_rainfall:.1f} mm could improve yield by {optimal_yield - predicted_yield:.2f} kg/ha")
        else:
            st.info("✅ Your rainfall is at the optimal level!")
    
    # ============================================================================
    # SECTION 10: FOOTER AND ADDITIONAL INFORMATION
    # ============================================================================
    
    st.markdown("---")
    st.markdown("""
    ### 📚 How to Use This Dashboard:
    
    1. **Adjust Parameters**: Use the sliders in the left sidebar to input different climate and soil conditions
    2. **View Prediction**: The predicted yield updates automatically as you change inputs
    3. **Analyze Factors**: Check the Feature Importance chart to see which variables matter most
    4. **Optimize Rainfall**: Use the Sensitivity Analysis to find the optimal rainfall for your conditions
    
    ### 🔬 About the Model:
    
    This dashboard uses **XGBoost (Extreme Gradient Boosting)**, a powerful machine learning algorithm that:
    - Learns complex patterns from historical cocoa yield data
    - Automatically tunes hyperparameters for best performance
    - Handles non-linear relationships between climate, soil, and yield
    
    ### 📊 Model Training Details:
    
    - **Training Data**: 80% of the dataset
    - **Testing Data**: 20% of the dataset
    - **Hyperparameter Tuning**: RandomizedSearchCV with 20 iterations
    - **Missing Values**: Automatically filled with median values
    
    ---
    
    💡 **Tip**: Try extreme values on the sliders to understand how each factor affects yield!
    
    Built with ❤️ using Streamlit, XGBoost, and Plotly
    """)

# ============================================================================
# SECTION 11: RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
