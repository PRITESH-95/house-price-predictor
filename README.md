<div align="center">

# House Price Predictor

**Predict Mumbai house prices with machine learning.**
A TLPBL project for TY AIML.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

</div>

---

## Overview

This project trains a regression model with scikit-learn on a Mumbai house prices dataset and uses it to estimate the price of a property from its details.

Predictions are estimates for learning purposes, not property valuations.

## Quick start (Windows, PowerShell)

```powershell
git clone https://github.com/PRITESH-95/house-price-predictor.git
cd house-price-predictor
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Train and evaluate the model:

```powershell
python house_price_prediction.py
```

Run the interactive predictor:

```powershell
python house_price_ui.py
```

## Project structure

```
house-price-predictor/
+-- Mumbai_House_Prices_csv/
|   +-- Mumbai House Prices.csv   # dataset
+-- house_price_prediction.py     # data prep, model training, evaluation
+-- house_price_ui.py             # interface for entering details and getting a price
+-- requirements.txt
+-- README.md
```

## Tech stack

| Purpose | Tool |
|---|---|
| Data handling | pandas, NumPy |
| Modelling | scikit-learn |
| Language | Python |

## Author

Built by [PRITESH-95](https://github.com/PRITESH-95), DBATU, AIML.
