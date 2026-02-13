# 🍽️ Restaurant Demand Forecasting & Waste Reduction AI

> An end-to-end, production-grade Machine Learning system designed to predict daily customer footfall and item-wise sales, enabling optimized inventory management and significant food waste reduction.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Problem Statement

Restaurants face a critical challenge in balancing inventory with fluctuating demand.

- **Over-preparation** leads to **food waste** and financial loss.
- **Under-preparation** results in **lost revenue** and unhappy customers.
- **Manual forecasting** is inefficient and error-prone due to complex variables like weather, holidays, and seasonality.

## 🚀 Solution Overview
This project implements a **Multi-Output Regression** system that forecasts 6 key metrics simultaneously:
1. **Total Customers Served**
2. **LPG Usage (kg)**
3. **Dosa Sales**
4. **Idly Sales**
5. **Vada Sales**
6. **Puri Sales**

The solution features a modular architecture, hyperparameter-tuned models, and an interactive **Streamlit Dashboard** for real-time analytics and scenario simulation.

## ✨ Key Features
- **📊 Modular Architecture**: Clean separation of concerns (`DataIngestion`, `DataTransformation`, `ModelTrainer`).
- **🤖 Advanced Algorithms**: Compares **Random Forest**, **XGBoost**, **Gradient Boosting**, **Ridge**, and **Lasso** using **GridSearchCV** (5-fold CV).
- **📈 Comprehensive Metrics**: Evaluates models on **R2**, **RMSE**, **MAE**, **MAPE**, and **Explained Variance**.
- **🔍 Explainability**: Integrated **SHAP (SHapley Additive exPlanations)** to interpret model predictions.
- **📱 Interactive Dashboard**: 
    - **Scenario Simulation**: "What if it rains heavily on a holiday?"
    - **Business KPIs**: Track Efficiency (Customers/LPG) and Estimated Revenue.
    - **Model Performance**: Compare models via interactive charts.
- **☁️ Production Ready**: robust logging, custom exception handling, and artifact management (`JSON` metadata).

## 🛠️ Tech Stack
- **Language**: Python
- **Core Libraries**: Pandas, NumPy, Scikit-Learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, SHAP
- **Web Framework**: Streamlit
- **DevOps**: Logging, Exception Handling, Modular Pipelines

## 📂 Project Structure

```text
Food_Waste_Reduction/
├── artifacts/              # Generated models and metadata
│   ├── model.pkl           # Best trained model
│   ├── preprocessing.pkl   # Feature transformation pipeline
│   ├── model_performance.json
│   └── model_metadata.json
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code
│   ├── components/         # Core ML components (Ingestion, Transformation, Trainer)
│   ├── pipeline/           # Orchestration pipelines (Training, Prediction)
│   ├── utils.py            # Utility functions
│   ├── logger.py           # Logging configuration
│   └── exception.py        # Custom exception handling
├── app.py                  # Streamlit Dashboard application
├── requirements.txt        # Project dependencies
├── setup.py                # Package setup
└── README.md               # Project documentation
```

## ⚙️ How to Run locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/restaurant-demand-forecasting.git
   cd Food_Waste_Reduction
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate # Linux/Mac
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**

   Run the end-to-end training pipeline to generate artifacts:

   ```bash
   python -m src.pipeline.training_pipeline
   ```

5. **Launch the Dashboard**

   ```bash
   streamlit run app.py
   ```

## 📊 Results & Performance

The best performing model (typically **Gradient Boosting** or **Random Forest**) is automatically selected based on Test R2 Score.
- **R2 Score**: ~0.56 (Baseline) -> Improved via Tuning.
- **RMSE**: Optimized for minimal error in sales counts.

## 🔮 Future Improvements

- [ ] Integrate **Weather API** for real-time automated inputs.

---

**Author**: Sambhav Surthi
