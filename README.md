# Heart Disease Prediction System

A machine-learning project that explores heart-disease classification
from patient health and lifestyle attributes. It includes a Jupyter
Notebook for data exploration and model training, a saved Random Forest
model, and a Streamlit web application with dashboards, visualizations,
and an interactive prediction form.

> **Important medical disclaimer:** This project is for educational and
> demonstration purposes only. It is not a medical device and must not
> be used to diagnose, treat, or rule out heart disease. Any health
> concern should be discussed with a qualified healthcare professional.

## Features

-   **Home:** Project introduction, dataset summary metrics, and a
    preview of the data.
-   **Dashboard:** Visual summaries of heart-disease labels, age,
    gender, cholesterol, blood pressure, and heart rate.
-   **Visualizations:** Charts for smoking, diabetes, alcohol intake,
    chest-pain type, exercise hours, blood sugar, target distribution,
    and selected numerical features.
-   **Prediction:** A Streamlit form that collects patient attributes
    and passes them to the saved classification model.
-   **Model exploration:** A notebook that performs exploratory data
    analysis, encodes categorical columns, trains multiple classifiers,
    and evaluates their results.

## Technology Stack

-   Python
-   Streamlit
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Joblib
-   Jupyter Notebook

## Repository Structure

``` text
Heart-Disease-Prediction/
├── app.py                       # Streamlit web application
├── data.csv                     # Dataset used by the notebook and app
├── model.pkl                    # Saved Random Forest classifier
├── Heart_disease_prection.ipynb # EDA, preprocessing, training and evaluation
└── README.md                    # Project documentation
```

If your Python file has a different name, such as `app(1).py`, rename it
to `app.py` or use that filename in the run command.

## Dataset

The supplied `data.csv` contains **1,000 records**, **15 input
features**, and one target column named `Heart Disease`.

### Input features

  Feature                   Description
  ------------------------- -----------------------------------
  Age                       Patient age
  Gender                    Gender category in the dataset
  Cholesterol               Cholesterol measurement
  Blood Pressure            Blood pressure measurement
  Heart Rate                Heart rate measurement
  Smoking                   Smoking category
  Alcohol Intake            Alcohol-intake category
  Exercise Hours            Exercise hours
  Family History            Family history indicator
  Diabetes                  Diabetes indicator
  Obesity                   Obesity indicator
  Stress Level              Stress-level value
  Blood Sugar               Blood-sugar measurement
  Exercise Induced Angina   Exercise-induced angina indicator
  Chest Pain Type           Chest-pain category

**Target:** `Heart Disease` (0 or 1 in the supplied dataset).

The notebook reports 340 missing values in `Alcohol Intake` and no
duplicate rows. The notebook converts object-type columns to numeric
values using `LabelEncoder`.

## Machine Learning Workflow

The notebook follows these steps:

1.  Load the CSV dataset with Pandas.
2.  Inspect dataset dimensions, column types, missing values, duplicate
    rows, descriptive statistics, and target distribution.
3.  Explore the data using count plots, histograms, a correlation
    heatmap, and a box plot.
4.  Convert categorical columns to numeric values with `LabelEncoder`.
5.  Separate the input features (`X`) from the target (`y`).
6.  Split the data into training and testing sets using an 80/20 split
    and `random_state=42`.
7.  Train and evaluate three classification algorithms:
    -   Logistic Regression
    -   Decision Tree Classifier
    -   Random Forest Classifier (`n_estimators=200`, `random_state=42`)
8.  Evaluate the Random Forest model with a classification report and
    confusion matrix, inspect feature importances, and save the model as
    `model.pkl` using Joblib.

### Results recorded in the notebook

  Model                   Accuracy on the notebook test split
  --------------------- -------------------------------------
  Logistic Regression                                   80.5%
  Decision Tree                                          100%
  Random Forest                                          100%

These are the results printed by the supplied notebook for its single
train/test split. A 100% score should **not** be interpreted as proof of
real-world accuracy. Further validation (for example, cross-validation,
an independent dataset, and checks for overfitting and data leakage) is
needed before drawing conclusions.

## Installation

Use Python 3.9 or later (a compatible environment is recommended).

1.  Clone or download this repository.
2.  Open a terminal in the project folder.
3.  Create and activate a virtual environment (recommended):

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

4.  Install the dependencies:

``` bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib notebook
```

## Run the Streamlit App

Make sure `app.py`, `data.csv`, and `model.pkl` are in the same project
directory, then run:

``` bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) that
you can open in your browser.

## Run the Jupyter Notebook

``` bash
jupyter notebook
```

Open `Heart_disease_prection.ipynb` and run the cells in order. The
notebook saves the trained Random Forest classifier to `model.pkl`.

## Important Implementation Notes

The current files should be checked before using the prediction output:

-   **Categorical encoding must match:** During training, the notebook
    applies a separate `LabelEncoder` to each object column. The
    encoders are not saved. In the Streamlit form, the app manually maps
    some fields to 0/1 and chest-pain options to 0--3. Those manual
    mappings are not guaranteed to match the encodings learned in the
    notebook. Save and reuse the fitted encoders (ideally as part of a
    Scikit-learn `Pipeline` or preprocessing object) so training and
    inference use exactly the same transformations.
-   **Missing alcohol-intake values:** The dataset has 340 missing
    values in `Alcohol Intake`. The notebook converts missing values to
    strings during label encoding rather than applying an explicit
    missing-value imputation strategy. Review and document an
    appropriate missing-data treatment before retraining.
-   **Model compatibility:** `model.pkl` was serialized with a
    particular Scikit-learn version. Loading it with a different version
    can produce compatibility warnings or errors. Use the same
    compatible Scikit-learn version for training and inference, and only
    load pickle/joblib files from trusted sources.
-   **Prediction error handling:** The app should verify that the model
    loaded successfully before calling `model.predict()`.

These points are important for reproducible results and reliable
application behavior.

## Future Improvements

-   Build and save a complete preprocessing-and-model pipeline.
-   Handle missing values explicitly and consistently.
-   Use stratified cross-validation and evaluate precision, recall,
    F1-score, ROC-AUC, and a confusion matrix.
-   Validate the model on an independent, representative dataset.
-   Add input validation and clearer error handling to the Streamlit
    app.
-   Improve accessibility and mobile responsiveness.

## Disclaimer

Predictions from this educational project are not medical advice. The
model's output may be incorrect and must not be used as the basis for
medical decisions.
