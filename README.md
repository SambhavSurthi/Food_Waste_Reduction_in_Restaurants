# Restaurant Demand Forecasting

## Overview
This project predicts restaurant demand (Total Customers, LPG Usage, and Item Sales) using machine learning.

## Structure
- `artifacts/`: Stores models and preprocessors.
- `logs/`: Application logs.
- `notebooks/`: EDA and experiments.
- `src/`: Source code.
- `app.py`: Streamlit application.

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run EDA: `jupyter notebook notebooks/EDA.ipynb`
3. Train model: `python src/pipeline/training_pipeline.py`
4. Run App: `streamlit run app.py`
