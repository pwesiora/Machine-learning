import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
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


fig, (ax, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7)
ax.set_title('Raw data')
ax.boxplot(res)
ax.set_xticklabels(names)

###PCA###

features = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.99, whiten=True)
features_pca=pca.fit_transform(features)

test_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_pca, Y, test_size=test_size, random_state=seed)

print("\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("PCA:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
print("Original number of features:" + str(features.shape[1]))
print("New number of features:" + str(features_pca.shape[1]))

ax2.set_title('PCA')
ax2.boxplot(res)
ax2.set_xticklabels(names)


###KPCA###

#features = StandardScaler().fit_transform(X)
kpca = KernelPCA(kernel="rbf", gamma = 15, n_components=1)
features_kpca = kpca.fit_transform(X)

test_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_kpca, Y, test_size=test_size, random_state=seed)

print("\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("KPCA:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
print("Original number of features:" + str(features.shape[1]))
print("New number of features:" + str(features_kpca.shape[1]))

ax3.set_title('KPCA')
ax3.boxplot(res)
ax3.set_xticklabels(names)

###NMF###

# features = StandardScaler().fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(X, Y).transform(X)

test_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_lda, Y, test_size=test_size, random_state=seed)

print("\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("LDA:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
print("Original number of features:" + str(features.shape[1]))
print("New number of features:" + str(features_lda.shape[1]))

ax4.set_title('LDA')
ax4.boxplot(res)
ax4.set_xticklabels(names)

###NMF###

features = StandardScaler().fit_transform(X)
for i in features:
    i[i<0] = 0
    #print(i)

nmf = NMF(n_components=5, random_state = seed)
features_nmf = nmf.fit_transform(features)

test_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_nmf, Y, test_size=test_size, random_state=seed)

print("\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("NMF:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
print("Original number of features:" + str(features.shape[1]))
print("New number of features:" + str(features_nmf.shape[1]))

ax5.set_title('NMF')
ax5.boxplot(res)
ax5.set_xticklabels(names)
#plt.show()

###TSVD###

features = StandardScaler().fit_transform(X)
features_sparse = csr_matrix(features)
tsvd = TruncatedSVD(n_components=6)
features_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

test_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_tsvd, Y, test_size=test_size, random_state=seed)

print("\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("TSVD:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
print("Original number of features:" + str(features.shape[1]))
print("New number of features:" + str(features_tsvd.shape[1]))

ax6.set_title('TSVD')
ax6.boxplot(res)
ax6.set_xticklabels(names)


###KBEST###

fvalue_selector = SelectPercentile(f_classif, percentile=70)
features_kbest = fvalue_selector.fit_transform(X, Y)

test_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_kbest, Y, test_size=test_size, random_state=seed)

print("\n")
models = []
models.append(('KNeigh', KNeighborsClassifier()))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

res = []
names = []
print("KBEST:")
for name, model in models:
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kf, scoring=scoring)
    res.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s: %f" % (name, cv_results.mean())
    print(msg)
print("Original number of features:" + str(features.shape[1]))
print("New number of features:" + str(features_kbest.shape[1]))

ax7.set_title('KBEST')
ax7.boxplot(res)
ax7.set_xticklabels(names)
plt.show()


###RFECV###

X = array[:, 1:14]
Y = array[:, 0]
ols = linear_model.LinearRegression()
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(X, Y)
rfecv.transform(X)
print(rfecv.n_features_)