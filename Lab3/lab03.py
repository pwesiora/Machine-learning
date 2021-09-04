import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Iris dataset
# load dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pandas.read_csv(url, names=names)
#
# train, test split
# array = dataset.values
# X = array[:, 1:4]
# Y = array[:, 4]
# test_size = 0.20
# seed = 43434343
# fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
# cn = ['setosa', 'versicolor', 'virginica']

# Wine dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids'
    , 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines'
    , 'Proline']
dataset = pandas.read_csv(url, names=names)

print(dataset.head())
print(dataset.tail())

array = dataset.values
X = array[:, 1:14]
Y = array[:, 0]
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
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(res)
ax.set_xticklabels(names)
plt.show()

KNeigh = KNeighborsClassifier()
KNeigh.fit(X_train, Y_train)
predictions = KNeigh.predict(X_test)
print("\n\nK-Neighbour Classifier")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))

cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_test)
print("Decision Tree Classifier")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))
plt.figure(figsize=(18, 10))
plot_tree(cart, feature_names=fn, class_names=cn, filled=True);
plt.show()

svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)
print("Support Vector Machines")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))
