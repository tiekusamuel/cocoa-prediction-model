# 🍫 Cocoa Yield Prediction Dashboard - Setup Guide

## Quick Start Guide for Beginners

This dashboard helps you predict cocoa yields in Ghana based on climate and soil conditions. Follow these simple steps to get started!

---

## 📋 Prerequisites

You need Python installed on your computer. Download it from [python.org](https://www.python.org/downloads/) if you don't have it yet.

---

## 🚀 Installation Steps

### Step 1: Install Required Libraries

Open your terminal/command prompt and run:

```bash
pip install streamlit pandas numpy plotly xgboost scikit-learn
```

This installs all the necessary libraries:
- **streamlit**: Creates the web interface
- **pandas**: Handles data
- **numpy**: Mathematical operations
- **plotly**: Interactive charts
- **xgboost**: Machine learning model
- **scikit-learn**: Model training tools

### Step 2: Prepare Your Data

Make sure you have a file called `cocoa_data.csv` in the same folder as the dashboard script.

**Required columns in your CSV:**
- `rainfall` (mm)
- `temp_min` (°C)
- `temp_max` (°C)
- `solar_radiation` (MJ/m²)
- `soil_ph`
- `phosphorus` (ppm)
- `nitrogen` (ppm)
- `yield_kg_ha` (target variable - cocoa yield in kg per hectare)

**Example CSV structure:**
```csv
rainfall,temp_min,temp_max,solar_radiation,soil_ph,phosphorus,nitrogen,yield_kg_ha
1200,22.5,30.2,18.5,6.2,45,120,850
1350,21.8,29.8,19.2,6.5,52,135,920
...
```

### Step 3: Run the Dashboard

Navigate to the folder containing your script and CSV file, then run:

```bash
streamlit run cocoa_yield_dashboard.py
```

The dashboard will automatically open in your web browser at `http:/tie/localhost:8501`

---

## 🎯 How to Use the Dashboard

1. **Adjust Sliders**: Use the sidebar on the left to change climate and soil parameters
2. **View Predictions**: See the predicted yield update instantly in the main panel
3. **Explore Charts**: 
   - Feature Importance shows which factors matter most
   - Sensitivity Analysis shows how rainfall affects yield
4. **Optimize**: Find the best conditions for maximum yield

---

## 🔧 Troubleshooting

### Problem: "FileNotFoundError: cocoa_data.csv"
**Solution**: Make sure your CSV file is in the same folder as the Python script

### Problem: "ModuleNotFoundError"
**Solution**: Install the missing library using `pip install <library-name>`

### Problem: Dashboard won't open
**Solution**: Check if another program is using port 8501, or manually specify a port:
```bash
streamlit run cocoa_yield_dashboard.py --server.port 8502
```

---

## 📊 Understanding the Results

### Prediction Metrics
- **Predicted Yield**: Your simulated yield in kg/ha
- **Dataset Average**: Typical yield from the training data
- **Percentile Rank**: How your prediction compares to historical data

### Model Performance
- **R² Score**: How well the model fits (closer to 1.0 is better)
- **RMSE**: Average prediction error in kg/ha (lower is better)
- **MAE**: Mean absolute error (lower is better)

### Feature Importance
- Shows which variables have the biggest impact on yield
- Higher importance = stronger influence on predictions

### Sensitivity Analysis
- Shows how yield changes when rainfall varies
- Red star = your current selection
- Use this to find optimal rainfall levels

---

## 💡 Tips for Best Results

1. **Data Quality**: Use at least 100 data points for reliable predictions
2. **Realistic Ranges**: Keep slider values within typical Ghana conditions
3. **Experiment**: Try extreme values to understand limits
4. **Compare Scenarios**: Test different combinations to optimize yields

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **XGBoost Guide**: https://xgboost.readthedocs.io
- **Cocoa Growing in Ghana**: Research local agricultural extension services

---

## 🆘 Need Help?

If you encounter issues:
1. Check that all libraries are installed correctly
2. Verify your CSV file format matches the requirements
3. Make sure Python version is 3.7 or higher
4. Try running in a fresh terminal/command prompt

---

## 📝 Customization Ideas

Want to extend the dashboard? Try:
- Adding more sensitivity charts for other variables
- Including weather forecast integration
- Adding historical yield trends
- Creating downloadable reports
- Comparing multiple scenarios side-by-side

---

**Happy Predicting! 🌾**

Built with ❤️ for Ghana's cocoa farmers and researchers
