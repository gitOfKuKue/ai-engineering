# Data Science Salary Prediction

Predict `salary_in_usd` from a job profile using the Kaggle-style dataset in `datasets/ds_salaries.csv`.

This is a **regression** problem, not classification. The target is a dollar amount, not a category.

## Dataset

- **607 rows**, **11 columns**, **no missing values**
- Years: 2020 (72), 2021 (217), 2022 (318)
- Target: `salary_in_usd` (mean about $112,298, median about $101,570, range $2,859–$600,000)

Do **not** predict `salary`. That column mixes currencies. Drop `salary` and `salary_currency` so they cannot leak into the target.

### Columns used as features

| Column | Role |
|---|---|
| `work_year` | numeric |
| `experience_level` | ordinal: EN < MI < SE < EX |
| `employment_type` | one-hot: FT, PT, CT, FL |
| `job_title` | grouped (rare titles → Other), then one-hot |
| `remote_ratio` | numeric: 0, 50, 100 |
| `company_location` | grouped, then one-hot |
| `company_size` | ordinal: S < M < L |

`employee_residence` is left out of the linear model because it is too similar to `company_location`.

## Project files

| File | What it does |
|---|---|
| `info_dataset.py` | Prints shape, dtypes, head, describe, missing values, unique counts |
| `scatter_plot.py` | Salary vs year, colored by experience |
| `train_test.py` | Time split and train/test scatter |
| `linear_regression_baseline.py` | Encodes features, fits the model, scores 2022, predicts from user input |
| `requirements.txt` | pandas, numpy, matplotlib, scikit-learn |

Run from the `Exercises` folder with the local venv:

```bash
source venv/bin/activate
python info_dataset.py
python scatter_plot.py
python train_test.py
python linear_regression_baseline.py
```

## Train / test split

Train on **2020–2021** (289 rows). Test on **2022** (318 rows).

This is closer to “future salary” than a random 80/20 split. The test set is larger than the train set because most rows are from 2022. 2022 pay is also higher (test mean about $124,522 vs train mean about $98,847), so this split is a hard test.

## Best model

**Gradient boosting** (`HistGradientBoostingRegressor`) is the best model for this dataset on the 2022 test set.

| Rank | Model | Why |
|---|---|---|
| 1 | Gradient boosting | Highest R² (**0.398**), lowest MAE (**$33,003**), lowest RMSE (**$45,076**) |
| 2 | Linear regression | Almost as accurate, and the coefficients are easy to explain |
| 3 | Random forest | Worst of the three; overfits 289 train rows and fails on 2022 |

Why gradient boosting wins:

- It can learn **non-linear** effects and **interactions** (for example senior + Data Analyst + US) that a straight line misses.
- It still works with a **small** table (607 rows).
- On a true future split it beat linear regression on **all three** scores, not just R².

The win is **small** (about +0.02 R² and −$800 MAE). Linear regression is still the right model in `linear_regression_baseline.py` because it is simpler to interpret. Use boosting if the goal is the best 2022 score; use linear regression if the goal is to explain *why* a salary is high or low.

Do **not** pick random forest here. It had the lowest R² (**0.119**) and the highest RMSE (**$54,532**).

## Model implemented in this project

The saved workflow uses **multiple linear regression**:

```python
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
```

The model exists in memory as `regr` while the script runs. It is trained from the CSV each time. There is no `model.pkl` on disk.

Encoding:

- Map `experience_level` and `company_size` to numbers because linear regression only uses numbers, and those columns have a real order.
- One-hot encode `employment_type`, `job_title`, and `company_location`.
- Fit encoders on train only. Job titles and locations with fewer than 10 train rows become `Other`.

Interactive prediction asks for **year**, **experience level**, **employment type**, and **job title**. Location, company size, and remote ratio are assumed as **US, medium company, fully remote**.

## Results (2022 test set)

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear regression (current model) | 0.378 | $33,847 | $45,814 |
| Gradient boosting | 0.398 | $33,003 | $45,076 |
| Random forest | 0.119 | $39,278 | $54,532 |
| Guess the 2020–2021 average | -0.195 | $48,521 | $63,513 |

Linear regression beats guessing the average. Gradient boosting is slightly better. Random forest overfits the small train set and is worse on 2022.

R² of **0.37** is modest: the model explains about one-third of salary differences. That is a fair baseline for noisy salary data, not strong enough for an exact offer amount.

### How to read the errors

- **MAE** is the typical miss (about $34k).
- **RMSE** is higher than MAE because a few large misses get squared.
- About **61%** of 2022 rows fall within ±$34k of the prediction.
- A typical interval around a prediction is **prediction ± MAE**.

Example: SE / FT / Data Analyst / 2022 predicted about **$119,032**. People with that same profile in the CSV have mean about **$113k** and range **$60k–$170k**. One row at $90,320 is a below-average person, not proof the model is broken. Linear regression predicts the **typical** salary for a profile, not one employee.

## What the coefficients say

Largest linear effects (approximate):

- US location: about **+$59k**
- India location: about **−$43k**
- Each experience step: about **+$30k**
- Data Analyst: lower than the average title
- `CT` / `FL` coefficients are large but based on very few rows, so they are noisy

## Can R² go much higher?

A little, not a lot, with this CSV and a fair 2022 test.

- Gradient boosting only adds about **+0.02 R²**.
- The same job title already pays very different amounts.
- Skills, company name, and education are missing.
- A realistic ceiling here is about **0.40–0.50**, not 0.8.

Training and testing on mixed years can look better, but that is not a future test.

## Conclusion

The most suitable model for this dataset is **gradient boosting** (`HistGradientBoostingRegressor`).

This CSV is a **small tabular** salary table with mixed numeric and categorical fields. Pay depends on **combinations** of features, not on one line: a senior Data Analyst in the US is not the same as a senior Data Analyst in India, and experience does not add the same dollar amount for every title. Gradient boosting can learn those interactions. Linear regression cannot, except as separate add-on coefficients. The table has only **607 rows**, so a neural net is a poor fit, and **random forest** overfit the 289 train rows (R² **0.119** on 2022).

On the fair future split (train 2020–2021, test 2022), gradient boosting was best on every score: R² **0.398**, MAE **$33,003**, RMSE **$45,076**. Linear regression was close (R² **0.378**, MAE **$33,847**, RMSE **$45,814**) and remains the model in `linear_regression_baseline.py` because the coefficients are easy to explain.

That does not make boosting a precise salary oracle. R² near **0.40** is still modest. The same job profile already pays **$60k–$170k**, and richer data (skills, company, education) is missing. Gradient boosting is the most suitable **among the models compared**, not a guarantee of a high score. Use it when the goal is the best 2022 prediction; keep linear regression when the goal is to explain the result. Report a **range** (prediction ± MAE), not only one number.
