import seaborn as sns
from IPython.display import display
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier

df = pd.read_csv("water_potability.csv")

# Cleaning Data From Null and duplicated Values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

display(df.shape)
display(df.head())

hm = sns.heatmap(data=df.corr(),annot=True)
# to show
plt.show()

x = df.drop(["Potability"], axis=1)
y = df["Potability"]
print("The shape of the data before removing outliers: {}".format(x.shape))
# Calculate Q1 (25th quantile of the data) for all features.
Q1 = x.quantile(0.25)

# Calculate Q3 (75th quantile of the data) for all features.
Q3 = x.quantile(0.75)

# Use the interquartile range to calculate an outlier step (1.5 times the interquartile range).
IQR = Q3 - Q1
step = 1.5 * IQR

# Remove the outliers from the dataset.
x_out = x[~((x < (Q1 - step)) |(x > (Q3 + step))).any(axis=1)]

# Join the features and the target after removing outliers.
preprocessed_data = x_out.join(y)
y_out = preprocessed_data[preprocessed_data.columns[-1]]

# Print data shape after removing outliers.
print("The shape of the data after removing outliers: {}".format(preprocessed_data.shape))

scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x_out)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_out, test_size=0.20, random_state=271)

tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
tree.fit(x_train, y_train)
prediction1 = tree.predict(x_test)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
prediction2 = knn.predict(x_test)

rf = RandomForestClassifier(random_state=11)
rf.fit(x_train, y_train)
prediction3 = rf.predict(x_test)

gaussian=GaussianNB()
gaussian.fit(x_train, y_train)
prediction4 = gaussian.predict(x_test)

print("AccuracyTree:",metrics.accuracy_score(y_test, prediction1))
print("AccuracyKNN:",metrics.accuracy_score(y_test, prediction2))
print("AccuracyRandomForest:",metrics.accuracy_score(y_test, prediction3))
print("AccuracyGaussian:",metrics.accuracy_score(y_test, prediction4))