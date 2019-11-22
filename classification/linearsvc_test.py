import ssl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, preprocessing, svm

from mlxtend.plotting import plot_decision_regions

ssl._create_default_https_context = ssl._create_unverified_context
df_wine_all = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)

colnums = []
colnames = []
while True:
    try:
        colnum = input("Please enter the column number:")
        colname = input("Please enter the column name:")
    except EOFError:
        break

    colnums.append(int(colnum))
    colnames.append(int(colname))
df_wine = df_wine_all[colnums]
df_wine.columns = colnames
# df_wine.columns = [u"class", u"color", u"proline"]

x = df_wine["color"]
y = df_wine["proline"]
z = df_wine["class"] - 1
plt.scatter(x, y, c=z)
plt.show

X = df_wine[["color", "proline"]]
sc = preprocessing.StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# loss='squared_hinge' #loss="hinge", loss="log"
clf_result = svm.LinearSVC(loss="hinge", C=1.0, class_weight="balanced", random_state=0)
clf_result.fit(X_std, z)

scores = model_selection.cross_val_score(clf_result, X_std, z, cv=10)
print("平均正解率 = ", scores.mean())
print("正解率の標準偏差 = ", scores.std())


X_train, X_test, train_label, test_label = model_selection.train_test_split(
    X_std, z, test_size=0.1, random_state=0
)
clf_result.fit(X_train, train_label)

pre = clf_result.predict(X_test)
ac_score = metrics.accuracy_score(test_label, pre)
print("正答率 = ", ac_score)

X_train_plot = np.vstack(X_train)
train_label_plot = np.hstack(train_label)
X_test_plot = np.vstack(X_test)
test_label_plot = np.hstack(test_label)
# plot_decision_regions(X_train_plot, train_label_plot, clf=clf_result, res=0.01)
plot_decision_regions(X_test_plot, test_label_plot, clf=clf_result, res=0.01, legend=2)

# predicted_label=clf_result.predict([1,-1])
# print("このテストデータのラベル = ", predicted_label)


print(clf_result.intercept_)
# coef[0]*x+coef[1]*y+intercept=0
print(clf_result.coef_)
