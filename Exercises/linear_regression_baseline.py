from pathlib import Path
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

df = pandas.read_csv(Path(__file__).parent / "datasets" / "ds_salaries.csv", index_col=0)

df["experience_level"] = df["experience_level"].map({"EN": 0, "MI": 1, "SE": 2, "EX": 3})
df["company_size"] = df["company_size"].map({"S": 0, "M": 1, "L": 2})

features = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "remote_ratio",
    "company_location",
    "company_size",
]
target = "salary_in_usd"
cat_cols = ["employment_type", "job_title", "company_location"]

train = df[df["work_year"] <= 2021].copy()
test = df[df["work_year"] == 2022].copy()

common_titles = train["job_title"].value_counts()
common_titles = common_titles[common_titles >= 10].index
train["job_title"] = train["job_title"].where(train["job_title"].isin(common_titles), "Other")
test["job_title"] = test["job_title"].where(test["job_title"].isin(common_titles), "Other")

common_locations = train["company_location"].value_counts()
common_locations = common_locations[common_locations >= 10].index
train["company_location"] = train["company_location"].where(
    train["company_location"].isin(common_locations), "Other"
)
test["company_location"] = test["company_location"].where(
    test["company_location"].isin(common_locations), "Other"
)

train_x = pandas.get_dummies(train[features], columns=cat_cols)
test_x = pandas.get_dummies(test[features], columns=cat_cols)
train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)

train_y = train[target]
test_y = test[target]

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
predictions = regr.predict(test_x)

print("Train size:", len(train_x))
print("Test size:", len(test_x))
print("MAE:", round(mean_absolute_error(test_y, predictions), 2))
print("RMSE:", round(root_mean_squared_error(test_y, predictions), 2))
print("R2:", round(r2_score(test_y, predictions), 3))

coefficients = pandas.Series(regr.coef_, index=train_x.columns)
print("\nTop coefficients:")
print(coefficients.reindex(coefficients.abs().sort_values(ascending=False).index).head(10))

print("\nPredict salary")
print("Experience level: EN, MI, SE, EX")
print("Employment type: FT, PT, CT, FL")
print("Known job titles:", ", ".join(common_titles))
work_year = int(input("Enter work year: "))
experience_level = input("Enter experience level: ").strip().upper()
employment_type = input("Enter employment type: ").strip().upper()
job_title = input("Enter job title: ").strip()

experience_map = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
valid_employment = {"FT", "PT", "CT", "FL"}
title_lookup = {title.lower(): title for title in df["job_title"].unique()}

if experience_level not in experience_map:
    print("Invalid experience level. Use EN, MI, SE, or EX.")
elif employment_type not in valid_employment:
    print("Invalid employment type. Use FT, PT, CT, or FL.")
elif job_title.lower() not in title_lookup:
    print("Unknown job title. Try one of:", ", ".join(common_titles))
else:
    job_title = title_lookup[job_title.lower()]
    future = pandas.DataFrame([{
        "work_year": work_year,
        "experience_level": experience_map[experience_level],
        "employment_type": employment_type,
        "job_title": job_title,
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": 1,
    }])
    future["job_title"] = future["job_title"].where(future["job_title"].isin(common_titles), "Other")
    future["company_location"] = future["company_location"].where(
        future["company_location"].isin(common_locations), "Other"
    )
    future_x = pandas.get_dummies(future[features], columns=cat_cols)
    future_x = future_x.reindex(columns=train_x.columns, fill_value=0)
    predicted_salary = round(regr.predict(future_x)[0], 2)
    print("\nAssumed: US, medium company, fully remote")
    print("Predicted salary (USD):", predicted_salary)

plt.scatter(test_y, predictions, alpha=0.6, label="2022 predictions")
limit_min = min(test_y.min(), predictions.min())
limit_max = max(test_y.max(), predictions.max())
plt.plot([limit_min, limit_max], [limit_min, limit_max], color="red", label="Perfect fit")
plt.xlabel("Actual Salary (USD)")
plt.ylabel("Predicted Salary (USD)")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()
