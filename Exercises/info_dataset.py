from pathlib import Path
import pandas

df = pandas.read_csv(Path(__file__).parent / "datasets" / "ds_salaries.csv", index_col=0)

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nData types:")
print(df.dtypes)
print("\nInfo:")
df.info()
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe(include="all"))
print("\nMissing values:")
print(df.isnull().sum())
print("\nUnique values per column:")
print(df.nunique())

print("\nwork_year and salary_in_usd:")
print(df[["work_year", "salary_in_usd"]])