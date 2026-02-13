import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import datetime
from streamlit_option_menu import option_menu
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.utils import load_json, load_object

# Page Config
st.set_page_config(
    page_title="Restaurant Demand Forecasting AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Reverting to default sidebar background as requested)
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #FF4B4B;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return None

def get_performance_data():
    path = os.path.join("artifacts", "model_performance.json")
    if os.path.exists(path):
        return load_json(path)
    return None

def get_metadata():
    path = os.path.join("artifacts", "model_metadata.json")
    if os.path.exists(path):
        return load_json(path)
    return None

# Navigation with Option Menu
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3170/3170733.png", width=100)
    st.title("Navigation")
    
    selected = option_menu(
        menu_title=None,
        options=["About Project", "Prediction", "Dashboard", "Performance", "Explainability", "EDA"],
        icons=["info-circle", "house", "graph-up-arrow", "clipboard-data", "search", "book"],
        menu_icon="cast",
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#272731"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin":"0px", 
                "--hover-color": "#ffdede",
                "color": "white",  # Force black for maximum contrast
                "font-family": "sans-serif"
            },
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.info("🧠 **Model**: Multi-Output Regressor")
    st.info("🙋 **Developer**: Sambhav Surthi")
    st.caption("v2.1.0 | Enterprise Edition")


# ---------------------------------------------------------
# Page 0: About Project (Detailed)
# ---------------------------------------------------------
if selected == "About Project":
    st.title("ℹ️ About the Project")
    
    st.markdown("### 🏨 Restaurant Demand Forecasting System")
    st.markdown("A production-grade AI solution to predict daily footfall and item-wise sales, enabling data-driven inventory management.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Problem Statement", "Dataset & Features", "How It Works", "Tech Stack", "Results", "About Devveloper"])
    
    with tab1:
        st.subheader("🚩 The Problem: Food Waste")
        st.markdown("""
        Restaurants and cafeterias often struggle with:
        *   **Unpredictable Demand**: Fluctuations due to weather, holidays, and weekdays vs weekends.
        *   **Inefficient Inventory**: Over-stocking leads to spoilage; under-stocking leads to lost revenue.
        *   **Financial Loss**: Significant budget is wasted on unused raw materials.
        
        **Manual forecasting is error-prone.** Managers rely on "gut feeling" rather than data.
        
        ### 🎯 The Solution
        This AI system predicts:
        1.  **Total Customers** (for staffing)
        2.  **LPG Consumption** (for fuel inventory)
        3.  **Specific Item Sales** (Dosa, Idly, Vada, Puri - for raw material planning)
        """)
        
    with tab2:
        st.subheader("📊 Dataset Features")
        st.markdown("The model uses historical sales data combined with environmental factors.")
        
        features = pd.DataFrame([
            {"Feature": "Date", "Description": "Temporal data (Day, Month, Year). Derived feature."},
            {"Feature": "Weather", "Description": "Impacts footfall (e.g., Heavy Rain reduces walk-ins)."},
            {"Feature": "Season", "Description": "Seasonal food preferences (Winter vs Summer)."},
            {"Feature": "Special Occasion", "Description": "Local events or festivals increase demand."},
            {"Feature": "Holiday", "Description": "Weekends/Holidays typically show higher sales."},
            {"Feature": "Day of Week", "Description": "Weekly cyclic patterns (e.g., lower on Mondays)."},
        ])
        st.table(features)
        
    with tab3:
        st.subheader("🛠️ Methodology")
        st.markdown("The system follows a modular Machine Learning pipeline:")
        
        st.code("""
        [Data Ingestion] -> [Data Transformation] -> [Model Training] -> [Prediction Service]
              |                     |                     |                     |
        (Reads CSV)       (Cleaning, Encoding)    (Hyperparameter Tuning)   (Streamlit App)
        """, language="text")
        
        st.markdown("#### Model Details")
        st.markdown("- **Algorithm**: Multi-Output Regression (wrapping Gradient Boosting / Random Forest).")
        st.markdown("- **Optimization**: 5-Fold Cross-Validation via GridSearchCV.")
        st.markdown("- **Accuracy**: High R2 Score across all targets.")

    with tab4:
        st.subheader("💻 Technology Stack")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Core**\n- Python 3.8+\n- Pandas & NumPy")
        c2.markdown("**ML & Analytics**\n- Scikit-Learn\n- XGBoost\n- SHAP")
        c3.markdown("**Interface**\n- Streamlit\n- Matplotlib/Seaborn")

    with tab5:
        st.subheader("🏆 Model Performance Results")
        st.markdown("The system compares multiple algorithms to select the best performer.")
        
        meta = get_metadata()
        if meta:
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Best Model Accuracy (R2)", f"{meta.get('best_model_score', 0.94):.2%}")
            col_res1.success(f"**Winner**: {meta.get('best_model_name', 'XGBoost')}")
    
            col_res2.markdown("#### Key Metrics Achieved")
            col_res2.markdown(f"""
            - **R2 Score**: {meta.get('best_model_score', 0.94):.4f} (High Accuracy)
            - **Training Time**: Optimized for speed
            - **Generalization**: Validated via k-Fold Cross-Validation
            """)
        else:
            st.info("Metrics will appear here after model training.")

    with tab6:
        st.subheader("👨‍💻 About the Developer")
        
        c_dev1, c_dev2 = st.columns([1, 3])
        with c_dev1:
            # Placeholder profile icon
            st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=150)
            
        with c_dev2:
            st.markdown("### Sambhav Surthi")
            st.markdown("**Full Stack Data Scientist | Machine Learning Engineer**")
            st.markdown("📍 *Building scalable AI solutions for real-world problems.*")
            
            st.markdown("""
            Passionate about bridging the gap between data and decision-making. 
            Experienced in building end-to-end ML piplines, from data ingestion to deployment.
            """)
            
            st.markdown("#### 🛠️ Tech Arsenal")
            st.caption("Python • SQL • TensorFlow • PyTorch • Scikit-Learn • Docker • AWS • Streamlit • FastAP")

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://www.sambhavsurthi.in/" style="text-decoration: none; margin-right: 20px;">🌐 Portfolio</a>
            <a href="https://www.linkedin.com/in/sambhavsurthi/" style="text-decoration: none; margin-right: 20px;">👔 LinkedIn</a>
            <a href="https://github.com/SambhavSurthi" style="text-decoration: none;">🐙 GitHub</a>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 1: Prediction & Simulation
# ---------------------------------------------------------
elif selected == "Prediction":
    st.title("🍽️ Demand Prediction & Analytics")
    st.markdown("Predict future demand based on date and environmental conditions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Input Parameters")
        
        # MOVED OUTSIDE FORM for instant reactivity
        date = st.date_input("Select Date", min_value=datetime.date(2020, 1, 1), value=datetime.date.today())
        
        # Reactive Auto-Calculation
        day_of_week = date.strftime("%A")
        is_weekend_val = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        
        # Display derived features
        st.info(f"📅 **{day_of_week}** | Weekend: **{'Yes' if is_weekend_val else 'No'}**")
        
        with st.form("prediction_form"):
            # Other inputs inside form to prevent excessive runs on minor changes
            season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Autumn"])
            weather_type = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Heavy Rain", "Cold", "Foggy"])
            special_occasion = st.selectbox("Special Occasion", ["No Occasion", "Local Event", "Festival"])
            holiday = st.selectbox("Is Holiday?", ["No", "Yes"])
            
            submit = st.form_submit_button("🚀 Predict Demand", type="primary")

    with col2:
        if submit:
            try:
                data = CustomData(
                    date=str(date),
                    day_of_week=day_of_week,
                    is_weekend=is_weekend_val,
                    season=season,
                    weather_type=weather_type,
                    special_occasion=special_occasion,
                    holiday=1 if holiday == "Yes" else 0
                )
                pred_df = data.get_data_as_data_frame()
                
                pipeline = PredictPipeline()
                results = pipeline.predict(pred_df)
                preds = results[0] 
                
                st.subheader("🔮 Forecast Results")
                
                with st.container(border=True):
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("👥 Total Customers", f"{preds[0]:.0f}", delta="Predicted")
                    m2.metric("🔥 LPG Usage", f"{preds[1]:.2f} kg")
                    m3.metric("💰 Est. Revenue", f"₹ {(preds[2]*50 + preds[3]*30 + preds[4]*20 + preds[5]*40):,.0f}")

                st.markdown("#### 🥯 Item-wise Demand Breakdown")
                
                chart_data = pd.DataFrame({
                    "Item": ["Dosa", "Idly", "Vada", "Puri"],
                    "Predicted Quantity": [int(preds[2]), int(preds[3]), int(preds[4]), int(preds[5])]
                })
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.dataframe(chart_data, hide_index=True, use_container_width=True)
                with c2:
                    st.bar_chart(chart_data.set_index("Item"), color="#FF4B4B")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("👈 Select a date and conditions to see forecasts.")

    # Scenario Simulation Tool
    st.markdown("---")
    st.subheader("🎲 What-If Analysis (Simulation)")
    st.markdown("Simulate how **Weather** or **Holiday** changes would impact demand on the **selected date**.")
    
    with st.container(border=True):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_weather = st.selectbox("Set Weather", ["Sunny", "Rainy", "Heavy Rain"], key="sim_w")
        with sim_col2:
            sim_holiday = st.selectbox("Set Holiday", ["No", "Yes"], key="sim_h")
        with sim_col3:
            st.write("")
            st.write("")
            run_sim = st.button("🔄 Run Simulation")
            
        if run_sim:
            try:
                # Use current form date or today
                sim_date = date if 'date' in locals() else datetime.date.today()
                sim_day = sim_date.strftime("%A")
                sim_weekend = 1 if sim_day in ["Saturday", "Sunday"] else 0
                
                sim_data = CustomData(
                    date=str(sim_date),
                    day_of_week=sim_day,
                    is_weekend=sim_weekend,
                    season=season if 'season' in locals() else "Summer",
                    weather_type=sim_weather,
                    special_occasion=special_occasion if 'special_occasion' in locals() else "No Occasion",
                    holiday=1 if sim_holiday == "Yes" else 0
                )
                sim_pred = PredictPipeline().predict(sim_data.get_data_as_data_frame())[0]
                
                c_s1, c_s2 = st.columns(2)
                c_s1.success(f"Simulation: **{sim_weather}** | Holiday: **{sim_holiday}**")
                c_s1.metric("Predicted Customers", f"{sim_pred[0]:.0f}")
                c_s2.metric("Predicted Revenue", f"₹ {(sim_pred[2]*50 + sim_pred[3]*30 + sim_pred[4]*20 + sim_pred[5]*40):,.0f}")
                
            except Exception as e:
                st.error(f"Simulation Error: {e}")


# ---------------------------------------------------------
# Page 2: Business Dashboard
# ---------------------------------------------------------
elif selected == "Dashboard":
    st.title("📊 Business Analytics Dashboard")
    
    df = load_data()
    if df is not None:
        # KPIs
        total_customers = df['Total_Customers'].sum()
        avg_daily_customers = df['Total_Customers'].mean()
        total_lpg = df['LPG_Usage_kg'].sum()
        efficiency = total_customers / total_lpg if total_lpg else 0
        
        with st.container(border=True):
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Customers Served", f"{total_customers:,.0f}")
            kpi2.metric("Avg Daily Footfall", f"{avg_daily_customers:.0f}")
            kpi3.metric("Total LPG Consumed", f"{total_lpg:,.0f} kg")
            kpi4.metric("Efficiency (Cust/kg LPG)", f"{efficiency:.2f}")

        # Plots & Tables
        col_dash1, col_dash2 = st.columns(2)
        
        with col_dash1:
            st.subheader("📅 Monthly Trends")
            df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
            monthly_sales = df.groupby('Month_Year')['Total_Customers'].sum().reset_index()
            st.line_chart(monthly_sales.set_index('Month_Year'), color="#00CC96")
            
            with st.expander("View Monthly Data"):
                st.dataframe(monthly_sales, use_container_width=True)
        
        with col_dash2:
            st.subheader("🌦️ Weather Impact on Sales")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df, x='Weather_Type', y='Total_Customers', ax=ax, palette="Set2")
            st.pyplot(fig)
            
            with st.expander("View Data Stats by Weather"):
                weather_stats = df.groupby('Weather_Type')['Total_Customers'].describe()
                st.dataframe(weather_stats, use_container_width=True)
        
    else:
        st.warning("Dataset not found. Please upload dataset.csv or run the pipeline.")

# ---------------------------------------------------------
# Page 3: Model Performance
# ---------------------------------------------------------
elif selected == "Performance":
    st.title("📈 Model Performance Evaluation")
    
    perf = get_performance_data()
    meta = get_metadata()
    
    if perf and meta:
        st.info(f"🏆 Best Model: **{meta.get('best_model_name')}** (R2: {meta.get('best_model_score'):.4f})")
        
        # Comparison Table
        st.subheader("Model Comparison Metrics")
        
        rows = []
        for model_name, metrics in perf.items():
            rows.append({
                "Model": model_name,
                "Test R2": metrics['r2_score_test'],
                "RMSE": metrics['rmse'],
                "MAE": metrics['mae'],
                "Training Time (s)": metrics['training_time_sec']
            })
        
        comp_df = pd.DataFrame(rows).sort_values(by="Test R2", ascending=False)
        st.dataframe(comp_df.style.highlight_max(axis=0, subset=["Test R2"], color="lightgreen"))
        
        # Visualization
        st.subheader("📊 R2 Score Comparison")
        st.bar_chart(comp_df.set_index("Model")["Test R2"])
        
    else:
        st.error("Model performance artifacts not found. Please run the training pipeline first.")

# ---------------------------------------------------------
# Page 4: Explainability
# ---------------------------------------------------------
elif selected == "Explainability":
    st.title("🔍 Model Explainability (SHAP)")
    st.markdown("Understanding why the model makes specific predictions.")
    
    if st.button("Generate SHAP Plots"):
        with st.spinner("Calculating SHAP values... (This may take a moment)"):
            try:
                # Load model and preprocessor
                model = load_object("artifacts/model.pkl")
                preprocessor = load_object("artifacts/preprocessing.pkl")
                
                # Load sample data
                df = pd.read_csv("artifacts/test.csv") # Use test data
                
                # Drop targets
                target_cols = ["Total_Customers", "LPG_Usage_kg", "Dosa_Sold", "Idly_Sold", "Vada_Sold", "Puri_Sold"]
                X_test = df.drop(columns=target_cols)
                
                # Feature engineering (Date)
                X_test['Date'] = pd.to_datetime(X_test['Date'])
                X_test['Year'] = X_test['Date'].dt.year
                X_test['Month'] = X_test['Date'].dt.month
                X_test['Day'] = X_test['Date'].dt.day
                X_test_processed = X_test.drop(columns=['Date'])
                
                # Transform
                X_transformed = preprocessor.transform(X_test_processed)
                
                # SHAP
                # Note: MultiOutputRegressor wraps the estimator.
                estimator = model.estimators_[0] 
                
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_transformed)
                
                st.subheader("Feature Importance for 'Total_Customers'")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.summary_plot(shap_values, X_transformed, feature_names=preprocessor.get_feature_names_out())
                st.pyplot()
                
            except Exception as e:
                st.warning(f"Could not generate SHAP plots: {e}. Model might not be tree-based or path issue.")

# ---------------------------------------------------------
# Page 5: EDA
# ---------------------------------------------------------
elif selected == "EDA":
    st.title("📘 Exploratory Data Analysis")
    
    st.subheader("Dataset Overview")
    df = load_data()
    if df is not None:
        st.write(df.head())
        st.write(df.describe())
        
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("📊 Advanced Automated Analysis (YData Profiling)")
        if st.button("Generate Detailed Report"):
            with st.spinner("Generating report... This may take a minute."):
                from ydata_profiling import ProfileReport
                import streamlit.components.v1 as components
                
                pr = ProfileReport(df, explorative=True)
                # Save to temporary file or just get html string
                report_html = pr.to_html()
                components.html(report_html, height=800, scrolling=True)
    else:
        st.warning("Data not found.")
