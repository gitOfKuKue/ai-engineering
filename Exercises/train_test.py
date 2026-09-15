from pathlib import Path
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv(Path(__file__).parent / "datasets" / "ds_salaries.csv", index_col=0)

features = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]
target = "salary_in_usd"

X = df[features]
y = df[target]

train_x = X[df["work_year"] <= 2021]
train_y = y[df["work_year"] <= 2021]

test_x = X[df["work_year"] == 2022]
test_y = y[df["work_year"] == 2022]

print("Train years:", sorted(int(year) for year in train_x["work_year"].unique()))
print("Test years:", sorted(int(year) for year in test_x["work_year"].unique()))
print("Train size:", len(train_x))
print("Test size:", len(test_x))
print("Dropped columns: salary, salary_currency")

plt.scatter(train_x["work_year"], train_y, label="Train (2020-2021)", alpha=0.6)
plt.scatter(test_x["work_year"], test_y, label="Test (2022)", alpha=0.6)
plt.xlabel("Work Year")
plt.ylabel("Salary in USD")
plt.title("Train vs Test Split")
plt.legend()
plt.show()
