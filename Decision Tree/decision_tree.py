from numpy import mean, median
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("./Decision Tree/data.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

# Finding the midpoint of the ranks
ranks = sorted(df['Rank'].unique())
midpoints = [(ranks[i] + ranks[i + 1]) / 2 for i in range(len(ranks) - 1)]
print(midpoints)
print(dtree.tree_.threshold[0])

print(dtree)
print(tree.export_text(dtree, feature_names=features))

plt.show()