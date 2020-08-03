import sklearn
from sklearn .datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data=load_breast_cancer()
print(data.feature_names)
print(data.target_names)

print(data.DESCR)

x=data.data
y=data.target
print(x)
print(y)
model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

fet=pd.Series(model.feature_importances_)
fet.nlargest(8).plot(kind='barh')
plt.show()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train,y_train)

cls=KNeighborsClassifier()

cls.fit(x_train,y_train)

pre=cls.predict(x_test)

print(sklearn.metrics.accuracy_score(y_test,pre))
print(sklearn.metrics.classification_report(y_test,pre))
print(sklearn.metrics.confusion_matrix(y_test,pre))
print(sklearn.metrics.precision_score(y_test,pre))
print(sklearn.metrics.recall_score(y_test,pre))
print(sklearn.metrics.f1_score(y_test,pre))
print(sklearn.metrics.roc_auc_score(y_test,pre))
