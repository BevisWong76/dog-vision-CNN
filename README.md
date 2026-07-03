# Project Name

## Overview
This project tackles a complex ___ problem by 
This project provides an end-to-end Machine Learning solution to ___
By leveraging / analyzing ...
The pipeline evaluates ...


> For in-depth details for Exploratory Data Analysis (EDA), Model Selection and Tuning, and Results, please refer to the [Full Technical Report](report.md).

---

## Demo

![Streamlit App Demo](./assets/Demo.gif)

**Try the Live Interactive Web App:** [... on Streamlit Cloud](https://xxx.streamlit.app)

---

## Key Features & Highlights

* **Data Integrity:**

* **Feature Engineering:** 

* **Multi-Model Evaluation:**

* **Hyperparameter Optimization:**

* **Model Interpretability:** 

* **Interactive Web Interface:** 



---
## Key Results  

### 1. Model Performance Comparison


### 2. Champion Model Deep-Dive


### 3. Feature Importance & Model Interpretability



---

## Tech Stack

* **Language:** Python `3.12.11`
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, lightgbm
* **Visualization:** Matplotlib, Seaborn, statsmodels
* **Model Interpretability :** shap
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit

---

## Project Structure

```text
bulldozers-price-predictor/
├── assets/
│   └── demo.gif                        # Demonstration GIF for README
├── data/                               # Data set
├── bulldozers_price_regression.ipynb   # Complete Machine Learning pipeline
├── bulldozers_price_report.md          # Comprehensive technical report
├── app.py                              # Interactive Streamlit web application
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

The execution pipeline automatically generates and manages the following runtime directories:

```text
├── models/                             # Stores trained model files
└── plots/                              # Generated visualizations
    ├── EDA/                            # Exploratory Data Analysis plots
    ├── Model_Selection/                # Hyperparameter tuning and Evaluation Metrics
    └── Features/                       # Feature importance visualizations
```

---

## How to Run

First clone the repository:
```bash
git clone https://github.com/BevisWong76/bulldozers-price-predictor.git
cd bulldozers-price-predictor
```

You can then set up the project locally using either the standard Python `venv` or the ultra-fast `uv` package manager.

### Option 1: Using Standard Python `venv` (Traditional)

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using `uv` (Recommended for Speed)

`uv` is an extremely fast Python package installer and resolver written in Rust.

1. Install `uv` (if you haven't already):
```bash
pip install uv
```

2.  Create a virtual environment:

```bash
uv venv
```

3. Activate the virtual environment:
```bash
# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

 4. Install dependencies:
```bash
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### Run the Streamlit App

Once the dependencies are installed and the model artifacts are generated, launch the interactive web application:

```bash
streamlit run app.py
```
---

## Acknowledgements

* **Dataset:** 
* **Inspiration:** 

### Key Enhancements Beyond Baseline
* **Robust Pipeline Architecture:** 
* **Advanced Benchmarking & Tuning:**
* **Model Interpretability:** 
* **Interactive Web App:**