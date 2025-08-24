# Finding-Causal-Factor-behind-Sea-Level-Rise-in-the-Bay-of-Bengal-Region
1.Conducted sea level analysis and projection for 21.25°N, 87.75°E using time series modeling. 2. Detected heteroscedasticity and autocorrelation using Breusch-Pagan, White’s, and Durbin-Watson tests. 4.Identified top causal factors and their optimal lag for sea level rise using Pearl (DoWhy) and Granger causality methods.


Sea Level Rise Analysis and Causal Inference (1993–2035)
Location: 21.25°N, 87.75°E

Overview
This project investigates the drivers behind sea level rise at a specific coastal location in India, using a combination of statistical analysis, time series modeling, machine learning, and causal inference. The goal was to not only forecast sea level changes from 1993 to 2035 but also to identify the most influential and causal factors contributing to this rise.

Motivation
Rising sea levels pose significant risks to coastal communities, infrastructure, and ecosystems. Understanding not just the trends, but the underlying causes, is critical for effective climate adaptation and policy-making. This project aims to bridge the gap between predictive analytics and actionable insights by combining robust modeling with causal discovery.

Data Preparation
Data Sources:
Multi-decadal, multi-variable datasets including climate indicators (e.g., temperature, river discharge, greenhouse gas emissions) and economic variables.
Cleaning:
Removed irrelevant columns (e.g., 'year'), handled missing values, and ensured data consistency.
Scaling:
Used RobustScaler to normalize features, making the analysis robust to outliers.
Feature Engineering & Selection
Correlation Analysis:
Used heatmaps and NetworkX graphs to visualize and quantify feature correlations.
Removed highly correlated features (correlation > 0.9) to reduce multicollinearity, ensuring more reliable model interpretation and causal analysis.
Statistical Testing
Stationarity:
Applied the Augmented Dickey-Fuller (ADF) test to all variables.
Differenced non-stationary series to ensure validity for time series modeling.
Heteroscedasticity & Autocorrelation:
Used Breusch-Pagan and White’s tests to check for non-constant variance in residuals.
Performed Durbin-Watson test to detect autocorrelation.
These tests informed the choice of models (e.g., moving from OLS to ARIMA/SARIMAX and XGBoost).
Time Series Modeling
ARIMA/SARIMAX:
Modeled sea level as a function of its own past values and exogenous variables.
Used ACF and PACF plots to select optimal lag parameters.
SARIMAX allowed for inclusion of external drivers (e.g., emissions, temperature).
Validated model assumptions and residuals.
Machine Learning Modeling
XGBoost Regression:
Chosen for its robustness to heteroscedasticity and ability to capture non-linear relationships.
Hyperparameter tuning via RandomizedSearchCV and GridSearchCV (parameters: learning_rate, max_depth, n_estimators, min_child_weight, gamma, colsample_bytree).
Evaluated using Mean Squared Error (MSE) and R-squared (R²).
Feature importance extracted using XGBoost’s built-in metrics (gain, weight, cover).
Causal Inference
1. Pearl’s Causality (DoWhy Framework)
Theory:
Causal inference seeks to estimate the effect of a variable (treatment) on an outcome, controlling for confounders.
Used Directed Acyclic Graphs (DAGs) to represent assumed relationships.
Estimated Average Treatment Effect (ATE) for each variable using backdoor adjustment.
Implementation:
For each feature, constructed a causal model with all other features as confounders.
Identified the variable with the highest positive ATE as the strongest average causal driver of sea level rise.
2. Granger Causality
Theory:
Granger causality tests whether past values of one variable help predict another.
Useful for time-lagged relationships in time series data.
Implementation:
For each variable, tested Granger causality with sea level at multiple lags (up to 10 years).
Identified the variable and lag with the lowest significant p-value as the strongest time-based causal predictor.
Results
Heteroscedasticity was detected in linear models, but XGBoost and SARIMAX handled it robustly.
Top features by XGBoost importance:
(Insert your top features, e.g., 'sea_surface_temperature', 'river_discharge', etc.)
Most causal factor (DoWhy):
(Insert most causal factor, e.g., 'sea_surface_temperature') with the highest positive ATE.
Strongest Granger-causal factor:
(Insert factor, e.g., 'agricultural_mehtane_emissions') with optimal lag of (X) months/years.
Model performance:
Achieved R² of (insert value) and MSE of (insert value) on test data.
Projection:
Provided sea level projections for 21.25°N, 87.75°E from 1993 to 2035.
Why This Approach?
Combining statistical tests, time series models, machine learning, and causal inference provides a holistic understanding of both prediction and causation.
Feature selection and scaling ensure robust, interpretable models.
Causal inference (DoWhy, Granger) moves beyond correlation, identifying actionable drivers for policy and intervention.
How to Run
Clone the repository.
Install dependencies:
pip install -r requirements.txt
Place your data file as specified in the notebook.
Run Model.ipynb step by step for full analysis and results.
Keywords
Python, pandas, numpy, seaborn, matplotlib, NetworkX, statsmodels, scikit-learn, XGBoost, DoWhy, ARIMA, SARIMAX, RandomizedSearchCV, GridSearchCV, causal inference, feature importance, time series analysis, Granger causality, machine learning, hyperparameter tuning, data visualization, statistical testing, robust scaling, autocorrelation, heteroscedasticity, stationarity, ADF test, causal graphs, DAG, Average Treatment Effect (ATE), exogenous variables, environmental analytics, climate data science.

Acknowledgements
Open-source libraries: scikit-learn, statsmodels, xgboost, dowhy, matplotlib, seaborn, pandas, numpy.
World Bank and other climate data providers.
Contact
For questions or collaboration, please reach out via LinkedIn or GitHub Issues.

This project demonstrates a full end-to-end workflow for environmental time series analysis, robust prediction, and causal discovery.
