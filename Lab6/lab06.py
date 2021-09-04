import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, linear_model
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.sparse import csr_matrix

# Wine dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids'
    , 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines'
    , 'Proline']
dataset = pandas.read_csv(url, names=names)

# print(dataset.head())
# print(dataset.tail())

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
# models.append(('DTree', DecisionTreeClassifier()))
# models.append(('SVM', SVC()))

res = []
names = []
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)

    names.append(name)
    msg = "Cross_val accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)

KNeigh = KNeighborsClassifier()
KNeigh.fit(X_train, Y_train)
predictions = KNeigh.predict(X_test)
print("\n\nK-Neighbour Classifier")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))

classifier = DecisionTreeClassifier()
predictions = KNeigh.predict(X_test)
matrix = confusion_matrix(Y_test, predictions)
dataframe = pandas.DataFrame(matrix, index=cn, columns=cn)

sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.show()

dummy = DummyClassifier(strategy='uniform', random_state=seed)
dummy.fit(X_train, Y_train)
print("Dummy accuracy score: " + str(dummy.score(X_test, Y_test)))

print("The K-neighbour algorithm is has a better accuracy score than the dummy by: "
      + str(accuracy_score(Y_test, predictions) - dummy.score(X_test, Y_test)))

X = array[:, 1:14]
Y = array[:, 0]
Y[Y > 1] = 0
cn = ['1', '0']
res = []
names = []

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
scoring = 'accuracy'

for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)

    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
