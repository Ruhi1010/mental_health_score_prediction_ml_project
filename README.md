# Mental Health Score Prediction — Documentation

A machine learning web app that predicts a student's mental health score (0–10) from social media usage, study habits, sleep, and lifestyle inputs. Built with a **FastAPI** backend (trained scikit-learn model) and a vanilla **HTML/CSS/JS** frontend.

## Live Links

| Component | URL |
|---|---|
| Frontend (static site) | https://mental-health-score-prediction-ml-umah.onrender.com |
| Backend API | https://mental-health-score-prediction-ml.onrender.com |
| Interactive API docs (Swagger UI) | https://mental-health-score-prediction-ml.onrender.com/docs |


> Render free-tier services spin down after inactivity — the first request after idle time can take 30–60 seconds while the backend wakes up.

## Project Structure

```
mental_health_score_prediction_ml_project/
├── main.py                                          # FastAPI backend + /predict endpoint
├── mental_health_model.pkl                          # Trained scikit-learn model (joblib)
├── index.html                                       # Frontend form + result UI
├── script.js                                        # Frontend logic (calls the API)
├── style.css                                        # Frontend styling
├── requirements.txt                                 # Python dependencies
├── test.ipynb                                       # Model training / exploration notebook
└── Student Social Media And Mental Health Impact.csv # Training dataset
```

## Dataset

**File**: `Student Social Media And Mental Health Impact.csv` — 5,000 rows × 13 columns.

| Column | Type | Role |
|---|---|---|
| `Age` | numeric | feature |
| `Gender` | categorical (nominal) | feature |
| `Country` | categorical | raw — collapsed into `Grouped_Country` for modeling |
| `Academic_Level` | categorical | feature |
| `Most_Used_Platform` | categorical (nominal) | feature |
| `Purpose_Of_Use` | categorical (nominal) | feature |
| `Avg_Daily_Usage_Hours` | numeric | feature |
| `Daily_Unlocks` | numeric | feature |
| `Study_Hours` | numeric | feature — right-skewed, log-transformed |
| `Physical_Activity_Hours` | numeric | feature — clipped to ≥ 0 during cleaning |
| `Sleep_Hours_Per_Night` | numeric | feature |
| `Stress_Level` | categorical (ordinal: Low < Medium < High < Very High) | feature |
| `Mental_Health_Score` | numeric, 0–10 | **target** |

## Architecture

```
Browser (index.html + script.js)
        │
        │  fetch POST /predict  (JSON body)
        ▼
FastAPI backend (main.py)
        │
        │  pandas DataFrame → model.predict()
        ▼
mental_health_model.pkl (scikit-learn)
        │
        ▼
JSON response → rendered as a score + gauge in the browser
```

The frontend and backend are deployed as **two separate Render services** — a static site for the UI and a web service for the API.

---

## `test.ipynb` — Cell-by-Cell Explanation

This is the notebook that produces `mental_health_model.pkl`. Every code cell, in order:

### 1. Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
Standard data-science stack: `numpy`/`pandas` for data handling, `matplotlib`/`seaborn` for plotting during EDA.

### 2. Load and inspect the data
```python
df = pd.read_csv('Student Social Media And Mental Health Impact.csv')
df.shape        # (5000, 13)
df.head()       # first 5 rows
```
Loads the raw CSV into a DataFrame and does a first visual sanity check.

### 3. Basic data understanding
```python
df.isnull().sum()   # count of missing values per column
df.duplicated().sum()  # count of exact duplicate rows
df.info()            # dtypes + non-null counts
df.describe()         # mean/std/min/max/quartiles for numeric columns
```
Confirms there are no missing values to impute, but there **are** duplicate rows (handled later) and gives a first read on the numeric ranges (e.g. are there impossible negative hours anywhere).

### 4. Exploratory Data Analysis (EDA)
```python
sns.histplot(df['Mental_Health_Score'], kde=True)
```
Plots the distribution of the target variable — checks whether scores cluster, skew, or spread roughly normally across 0–10.

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```
Correlation heatmap across all numeric columns — a quick way to see which raw numeric features (sleep, usage hours, etc.) move together with `Mental_Health_Score` before any modeling.

```python
order = ['Low', 'Medium', 'High', 'Very High']
sns.boxenplot(x='Stress_Level', y='Mental_Health_Score', order=order, data=df)
```
Box-plot of the score grouped by stress level, in the natural Low→Very High order — this is what motivates treating `Stress_Level` as an **ordinal** feature later rather than one-hot encoding it.

```python
sns.scatterplot(x='Avg_Daily_Usage_Hours', y='Mental_Health_Score', data=df)
sns.scatterplot(x='Sleep_Hours_Per_Night', y='Mental_Health_Score', data=df)
```
Two scatter plots checking for a visible linear or non-linear relationship between screen time / sleep and the target.

```python
platform_counts = df['Most_Used_Platform'].value_counts()
sns.countplot(x='Most_Used_Platform', data=df, order=platform_counts.index)
```
Bar chart of how many students report each platform as their most-used one — informs the `Most_Used_Platform` list hardcoded into the Pydantic model in `main.py`.

### 5. Outlier check
```python
num_features = df.select_dtypes(include='number')
Q1 = num_features.quantile(0.25)
Q3 = num_features.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (num_features < lower_bound) | (num_features > upper_bound)
print(outliers.sum())
```
Standard IQR-based outlier flagging on every numeric column — used to decide what, if anything, needs clipping in the next step.

### 6. Data cleaning
```python
df = df.drop_duplicates()
```
Removes exact duplicate rows found earlier.

```python
df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)
```
Some rows had impossible negative activity-hour values; `clip(lower=0)` floors them at 0 instead of dropping the rows.

### 7. Skewness check
```python
num_cols = df.select_dtypes(include='number')
num_cols.skew()
```
Computes skewness per numeric column. Near-zero = roughly symmetric; positive = right-skewed (long tail toward high values). `Study_Hours` came out notably right-skewed, which is why it gets special treatment (log transform) in the preprocessing pipeline rather than being scaled directly like the other numeric columns.

### 8. Feature engineering — country grouping
```python
df['Country'].nunique()
top_countries = df['Country'].value_counts().index[:10].tolist()

def group_countries(country):
    if country in top_countries:
        return country
    else:
        return 'Other'

df['Grouped_Country'] = df['Country'].apply(group_countries)
```
`Country` has too many unique values to one-hot encode sensibly (it would explode the number of columns and most categories would have very few examples). This keeps the 10 most frequent country values as-is and buckets everything else into `"Other"`.

**Important detail**: `"Other"` is itself already a literal value present in the raw `Country` column (some respondents' country was recorded as `"Other"`), and it happens to be the single most common value in the whole dataset. So the top-10 list produced here is:
```python
['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']
```
This exact list (order aside) is duplicated by hand inside `main.py` as `top_countries`, so a live prediction request applies the identical bucketing rule the model was trained on.

### 9. Train/test split
```python
from sklearn.model_selection import train_test_split

skewed_col = ['Study_Hours']
other_numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
                       'Sleep_Hours_Per_Night', 'Physical_Activity_Hours']
ordinal_col = ['Stress_Level']
nominal_col = ['Gender', 'Most_Used_Platform', 'Grouped_Country', 'Purpose_Of_Use']

feature_col = skewed_col + other_numeric_cols + ordinal_col + nominal_col
X = df[feature_col]
y = df['Mental_Health_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```
Splits features into four groups by how they need to be preprocessed (this grouping is what the `ColumnTransformer` below acts on), then does a 70/30 train/test split with a fixed `random_state` for reproducibility. Note `Academic_Level` and raw `Country` are **not** in `feature_col` — `Academic_Level` isn't used by the trained model at all (even though `main.py`'s `StudentData` still collects and validates it), and raw `Country` was replaced by `Grouped_Country`.

### 10. Preprocessing pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

skewed_pipeline = Pipeline(steps=[
    ('log_transform', FunctionTransformer(np.log1p)),
    ('Scale', StandardScaler())
])

plain_numeric_pipeline = Pipeline(steps=[
    ('Scale', StandardScaler())
])

ordinal_pipeline = Pipeline(steps=[
    ('encode', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']]))
])

normal_pipeline = Pipeline(steps=[
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('skewed_pipeline', skewed_pipeline, skewed_col),
    ('plain_numeric_pipeline', plain_numeric_pipeline, other_numeric_cols),
    ('ordinal_pipeline', ordinal_pipeline, ordinal_col),
    ('normal_pipeline', normal_pipeline, nominal_col)
])
```
- `skewed_pipeline`: `log1p` (log(1+x), safe for zero values) compresses `Study_Hours`'s long tail, then scales it.
- `plain_numeric_pipeline`: just standard-scales the other numeric columns (mean 0, std 1) — necessary for Linear Regression, harmless for Random Forest.
- `ordinal_pipeline`: encodes `Stress_Level` as 0/1/2/3 in the explicit order Low→Very High, preserving the "more stress" ordering instead of treating the categories as unrelated.
- `normal_pipeline`: one-hot encodes the four true nominal columns (no natural order); `handle_unknown='ignore'` means a category never seen during training (e.g. a platform not in the training data) produces all-zero columns instead of crashing.
- `ColumnTransformer` glues all four together into one object that knows exactly which columns get which treatment.

### 11. Model building and comparison
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_preds_train = lr_pipeline.predict(X_train)
# ... R², MSE, MAE printed for both train and test
```
Baseline model: preprocessing + plain Linear Regression, fit on the training split, evaluated on both train and test sets (comparing the two catches overfitting).

```python
from sklearn.ensemble import RandomForestRegressor

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('random_forest', RandomForestRegressor(random_state=42))])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
rf_preds_train = rf_pipeline.predict(X_train)
# ... R², MSE, MAE printed
```
Same idea, but with a Random Forest using scikit-learn's default hyperparameters.

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'random_forest__n_estimators': [100, 200, 300],
    'random_forest__max_depth': [5, 10, 15],
    'random_forest__min_samples_split': [2, 5, 10],
    'random_forest__min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline, param_distributions=param_grid,
    n_iter=15, cv=5, scoring='r2', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
random_search.best_params_
```
Hyperparameter search over the Random Forest: 15 random combinations from the grid above, each evaluated with 5-fold cross-validation, scored by R². `random_search.best_estimator_` afterward is the best-performing pipeline found.

```python
rf_best_pipeline = random_search.best_estimator_
rf_best_preds = rf_best_pipeline.predict(X_test)
# ... R², MAE printed
```
Evaluates the *tuned* model on the held-out test set.

### 12. Model evaluation summary
```python
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_best_preds))
rf_tuned_preds = random_search.best_estimator_.predict(X_test)
rf_tuned_rmse = np.sqrt(mean_squared_error(y_test, rf_tuned_preds))
# ...
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Tuned Random Forest'],
    'R2': [...], 'Training R2': [...], 'MAE': [...], 'RMSE': [...]
})
print(results)
```
Builds a single comparison table (R², training R², MAE, RMSE for all three models) to decide which one is best.

### 13. Saving the model
```python
import joblib
joblib.dump(rf_pipeline, 'mental_health_model.pkl')
```
Serializes the fitted pipeline to disk. **This is the single most important line for deployment** — it's what `main.py` loads with `joblib.load('mental_health_model.pkl')` at startup.

> ⚠️ **This line saves `rf_pipeline` — the default-hyperparameter Random Forest — not `rf_best_pipeline`, the tuned model from `RandomizedSearchCV`.** Tuning was performed and evaluated above, but its result isn't what's actually deployed. If you want the tuned model live, change this line to `joblib.dump(rf_best_pipeline, 'mental_health_model.pkl')`, re-run the notebook, replace the `.pkl` file, and redeploy the backend.

---

## `main.py` — Line-by-Line Explanation

```python
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
```
Imports: `joblib` to load the saved model, `pandas` to build the single-row DataFrame the model expects, `FastAPI` for the web framework itself, `BaseModel`/`Field` from Pydantic for request validation, `Literal` to restrict a field to an exact set of allowed strings, and `CORSMiddleware` to allow browser requests from other origins.

```python
model = joblib.load('mental_health_model.pkl')
```
Loads the entire saved pipeline (preprocessing + Random Forest) **once**, when the server process starts — not on every request. This is why the server needs the `.pkl` file sitting next to `main.py` at startup, and why a broken/missing `.pkl` would crash the app immediately rather than only on the first `/predict` call.

```python
top_countries = ['Other', 'India', 'USA', 'Canada','Australia',
                 'UK', 'Germany', 'Turkey','Mexico', 'France']
```
Hand-copied from the notebook's `top_countries` (see `test.ipynb` step 8 above) so incoming requests get grouped into countries exactly the way the training data was. The order differs slightly from the notebook's own list (`Turkey`/`Mexico` swapped) but this list is only ever used for membership checks (`in`), so the order has no effect on behavior.

```python
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Creates the FastAPI app, then adds CORS middleware allowing requests from **any** origin, with any HTTP method and any headers. This is what lets `script.js`, served from a completely different domain (your Render static site, or `file://`, or `127.0.0.1`), successfully call this API without the browser blocking the request.

```python
class StudentData(BaseModel):
    Age                      : int = Field(..., ge=10, le=100)
    Gender                   : Literal['Male', 'Female']
    Country                  : str
    Academic_Level           : Literal['Undergraduate', 'Graduate', 'High School']
    Most_Used_Platform       : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
                                        'Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk',
                                        'VKontakte', 'WhatsApp', 'WeChat']
    Purpose_Of_Use           : Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours    : float = Field(..., ge=0, le=24)
    Daily_Unlocks            : int = Field(..., ge=0)
    Study_Hours              : float = Field(..., ge=0, le=24)
    Physical_Activity_Hours  : float = Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night    : float = Field(..., ge=0, le=24)
    Stress_Level             : Literal['Low', 'Medium', 'High', 'Very High']
```
This is the **request schema** — and the single most important contract in the whole project. Every field name here (`Age`, `Gender`, `Academic_Level`, …) must appear **exactly**, same case, in the JSON body of any POST to `/predict`, or FastAPI/Pydantic rejects the request with `422 Unprocessable Entity` before your route code ever runs. `Field(..., ge=X, le=Y)` means "required, and must be between X and Y inclusive." `Literal[...]` means "must be exactly one of these strings" — anything else (including a slightly different case or a typo) is rejected. This is exactly the schema `script.js`'s `API_FIELD_NAME` mapping and `toApiPayload()` function are built to match.

```python
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
```
The **response schema** — FastAPI uses this (via `response_model=PredictionResponse` on the route below) to both validate what the route returns and to auto-generate the response shape shown in `/docs`.

```python
@app.get("/")
def greet():
    return {"Hello! Welcome to the Mental Health Prediction API."}
```
A trivial health-check route at the root URL. Useful for confirming the server is up (and, on Render, for confirming the service has finished waking up from idle) without touching the model at all.

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):

    country_group = data.Country if data.Country in top_countries else 'Other'
```
The `/predict` route. FastAPI automatically parses the incoming JSON body, validates it against `StudentData`, and — if valid — hands you a fully-typed `data` object; if invalid, it never even reaches this line (a 422 is returned first). The first line inside reproduces the notebook's country-grouping logic: if the submitted country is one of the top 10 seen during training, keep it as-is; otherwise bucket it into `'Other'`.

```python
    input_row = pd.DataFrame([{
        'Age'                      : data.Age,
        'Gender'                   : data.Gender,
        'Country'                  : data.Country,
        'Academic_Level'           : data.Academic_Level,
        'Most_Used_Platform'       : data.Most_Used_Platform,
        'Purpose_Of_Use'           : data.Purpose_Of_Use,
        'Avg_Daily_Usage_Hours'    : data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks'            : data.Daily_Unlocks,
        'Study_Hours'              : data.Study_Hours,
        'Physical_Activity_Hours'  : data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night'    : data.Sleep_Hours_Per_Night,
        'Stress_Level'             : data.Stress_Level,
        'Grouped_Country'          : country_group
    }])
```
Builds a **one-row pandas DataFrame** from the validated request — `pd.DataFrame([{...}])` wraps a single dict in a list so pandas creates one row with these column names. This exactly mirrors the shape the model's `ColumnTransformer` was fit on during training (same column names it expects: `Study_Hours`, `Age`, `Avg_Daily_Usage_Hours`, …, `Grouped_Country`). Note `Country` and `Academic_Level` are included in this row but aren't actually consumed by the trained pipeline — `feature_col` in the notebook never included raw `Country` or `Academic_Level` — pandas simply ignores extra columns the `ColumnTransformer` wasn't told to use.

```python
    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))
```
`model.predict(input_row)` runs the *entire* saved pipeline in one call — the same `ColumnTransformer` steps (log-transform + scale, scale, ordinal-encode, one-hot-encode) are applied automatically before the Random Forest produces a prediction, because they were saved together as one `Pipeline` object. `model.predict()` returns a NumPy array (one prediction per input row); `[0]` grabs the single value since there's only one row. `round(float(prediction), 2)` converts the NumPy float type to a plain Python `float` (so it serializes cleanly to JSON) and rounds it to 2 decimal places. The result is wrapped in `PredictionResponse` and returned — FastAPI serializes it to the JSON your frontend receives.

---

## End-to-End Workflow

Tracing one prediction from click to result:

1. **User fills the form** in `index.html` and clicks submit. `script.js`'s submit handler fires and calls `e.preventDefault()` so the browser doesn't do a normal page reload.
2. **`collectPayload()`** reads every field via `FormData`, producing an internal object with lowercase keys (`age`, `gender`, …) matching the form's `id`/`name` attributes.
3. **`validate()`** re-checks the same numeric ranges and required-ness the backend will enforce. If anything fails, the relevant input is flagged red and the request never leaves the browser.
4. **`toApiPayload()`** converts the internal keys to the exact PascalCase names the FastAPI model expects (`age` → `Age`, `academic_level` → `Academic_Level`, …).
5. **`API_BASE` is resolved** based on `window.location` — local backend if the page itself is local, the Render backend URL otherwise.
6. **`fetch(`${API_BASE}/predict`, ...)`** sends the JSON payload as an HTTP POST.
7. **FastAPI receives the request** and Pydantic validates it against `StudentData`. Any mismatch (wrong field name, wrong type, out-of-range value, or an unlisted `Literal` option) returns `422` with a detailed per-field error list — `script.js`'s `applyServerValidationErrors()` maps these back onto the form.
8. **`predict()` runs**: country grouping → one-row DataFrame → `model.predict()` (which internally re-runs the exact preprocessing pipeline from training: log-transform/scale, scale, ordinal-encode, one-hot-encode) → Random Forest produces a numeric score.
9. **FastAPI returns JSON** `{"predicted_mental_health_score": 6.42}`.
10. **`script.js` receives the response**, checks the score is actually a number, then calls `renderResult()`: it clamps the score to 0–10, picks a qualitative "band" (strained / balanced / strong) via `bandFor()`, animates the gauge's SVG stroke offset, and swaps the UI into the "result" state.

---

## Frontend ↔ Backend Contract

`script.js` collects form input using lowercase, form-friendly keys (`age`, `gender`, …), then converts them to the API's exact field names before sending:

```js
const API_FIELD_NAME = {
  age: "Age",
  gender: "Gender",
  country: "Country",
  academic_level: "Academic_Level",
  most_used_platform: "Most_Used_Platform",
  purpose_of_use: "Purpose_Of_Use",
  avg_daily_usage_hours: "Avg_Daily_Usage_Hours",
  daily_unlocks: "Daily_Unlocks",
  study_hours: "Study_Hours",
  physical_activity_hours: "Physical_Activity_Hours",
  sleep_hours_per_night: "Sleep_Hours_Per_Night",
  stress_level: "Stress_Level",
};
```

It also selects the backend URL based on where the page itself is running:

```js
const LOCAL_HOSTS = ["127.0.0.1", "localhost", "", "[::1]"];
const isLocal =
  window.location.protocol === "file:" ||
  LOCAL_HOSTS.includes(window.location.hostname);
const API_BASE = isLocal
  ? "http://127.0.0.1:8000"
  : "https://mental-health-score-prediction-ml.onrender.com";
```

| Where the page is opened | `API_BASE` resolves to |
|---|---|
| Double-clicked `index.html` (`file://`) | `http://127.0.0.1:8000` |
| Served on `localhost` / `127.0.0.1` | `http://127.0.0.1:8000` |
| Render static site | `https://mental-health-score-prediction-ml.onrender.com` |

## API Reference

### `GET /`
Health check.
```json
{"Hello! Welcome to the Mental Health Prediction API."}
```

### `POST /predict`

Request body (all fields required):

| Field | Type | Constraints |
|---|---|---|
| `Age` | int | 10–100 |
| `Gender` | string | `"Male"` \| `"Female"` |
| `Country` | string | — |
| `Academic_Level` | string | `"Undergraduate"` \| `"Graduate"` \| `"High School"` |
| `Most_Used_Platform` | string | Facebook, LinkedIn, Instagram, Snapchat, Twitter, YouTube, TikTok, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat |
| `Purpose_Of_Use` | string | `"Networking"` \| `"Education"` \| `"Entertainment"` \| `"News"` |
| `Avg_Daily_Usage_Hours` | float | 0–24 |
| `Daily_Unlocks` | int | ≥ 0 |
| `Study_Hours` | float | 0–24 |
| `Physical_Activity_Hours` | float | 0–24 |
| `Sleep_Hours_Per_Night` | float | 0–24 |
| `Stress_Level` | string | `"Low"` \| `"Medium"` \| `"High"` \| `"Very High"` |

Example request:
```json
{
  "Age": 21,
  "Gender": "Female",
  "Country": "India",
  "Academic_Level": "Undergraduate",
  "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Entertainment",
  "Avg_Daily_Usage_Hours": 4.5,
  "Daily_Unlocks": 60,
  "Study_Hours": 3,
  "Physical_Activity_Hours": 1,
  "Sleep_Hours_Per_Night": 6.5,
  "Stress_Level": "Medium"
}
```

Example response:
```json
{"predicted_mental_health_score": 6.42}
```

Try it live: https://mental-health-score-prediction-ml.onrender.com/docs

## Running Locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the backend**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   API is now live at `http://127.0.0.1:8000`, docs at `http://127.0.0.1:8000/docs`.
3. **Open the frontend**
   Open `index.html` directly in a browser, or serve it (e.g. `python -m http.server 5500`). `script.js` auto-detects it's running locally and points requests at `http://127.0.0.1:8000`.

## Known Issue History

| Symptom | Cause | Fix |
|---|---|---|
| `/docs` predicts fine, but the web form never returns a score (422 error) | Frontend sent lowercase JSON keys (`age`, `academic_level`, …); Pydantic model requires exact-case keys (`Age`, `Academic_Level`, …) | Added `API_FIELD_NAME` mapping in `script.js` to convert keys before sending |
| Form works locally but fails after deploying to Render | `API_BASE` was hardcoded to `http://127.0.0.1:8000`, which on a deployed page points at the visitor's own machine, not the Render backend | `API_BASE` now switches based on `window.location` |

## Deployment Notes (Render)

- **Backend service** — start command should bind to Render's dynamic port:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
- **Frontend service** — deployed as a static site; redeploy it after any `script.js`/`index.html`/`style.css` change for the live site to reflect updates.
- CORS is open (`allow_origins=["*"]` in `main.py`), so the frontend can call the backend across origins without extra config.

## Tech Stack

- **Backend**: FastAPI, Pydantic, scikit-learn, pandas, joblib, uvicorn
- **Frontend**: vanilla HTML/CSS/JavaScript (no framework, no build step)
- **Hosting**: Render (separate static site + web service)