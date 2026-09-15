from pathlib import Path
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv(Path(__file__).parent / "datasets" / "ds_salaries.csv", index_col=0)

experience_colors = {
    "EN": "tab:green",
    "MI": "tab:blue",
    "SE": "tab:orange",
    "EX": "tab:red",
}

for level, color in experience_colors.items():
    subset = df[df["experience_level"] == level]
    plt.scatter(
        subset["work_year"],
        subset["salary_in_usd"],
        c=color,
        label=level,
        alpha=0.6,
    )

plt.xlabel("Work Year")
plt.ylabel("Salary in USD")
plt.title("Data Science Salaries by Year and Experience")
plt.legend(title="Experience")
plt.show()
