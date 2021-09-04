from random import random
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def fill_NaN(dataset, n):
    df = dataset.copy()
    for _ in  range(n):
        x = n.randint(1, 11)
        y = n.randint(0, 177)
        df.iloc[y, x] = None
    return df


# Wine dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids'
    , 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines'
    , 'Proline']
dataset = pandas.read_csv(url, names=names)

print(dataset.head())
print(dataset.tail())
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")


scaler = StandardScaler()
scaler.fit(dataset)
#print(scaler.transform(dataset))
# dataset=scaler.scale_(dataset)
# print(dataset.head())
# print(dataset.tail())
#
array = dataset.values
X = array[:, 1:14]
Y = array[:, 0]
print(X)
scaler.fit(X)

test_size = 0.20
seed = 43434343
fn = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids'
    , 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines'
    , 'Proline']
cn = ['1', '2', '3']

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
scoring = 'accuracy'

print("\n\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("Raw data:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)


fig, (ax, ax2) = plt.subplots(1, 2)
ax.set_title('Raw data')
ax.boxplot(res)
ax.set_xticklabels(names)

dataset=dataset.drop(columns=['Alcohol', 'Ash', 'Proline'])
dataset.dropna(thresh=2)
X =scaler.transform(X)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
res=[]

print("\nClean data:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)

ax2.set_title('Clean data')
ax2.set_xticklabels(names)
ax2.boxplot(res)
plt.show()


