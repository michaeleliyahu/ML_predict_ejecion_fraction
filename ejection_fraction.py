import seaborn as sns
import pandas as pd
import warnings as wr
import matplotlib.pyplot as plt
from sklearn.svm import SVC

wr.filterwarnings("ignore") #to ignore the warnings

df=pd.read_csv('C:/Users/Michael Eliyahu/Downloads/machine learning/final_froject/input/heart_failure_clinical_records_dataset.csv')
df.head()

# Create Classification version of target variable
df['goodquality'] = ['bad' if x <= 40 else  'good' for x in df['ejection_fraction']]
# Separate feature variables and target variable
X = df.drop(['ejection_fraction','goodquality'], axis = 1)
y = df['goodquality']

print(df['goodquality'].value_counts())

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


#X = df.iloc[:, :-1]
##X = df[['ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time','age']]
#y = df['DEATH_EVENT']

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
X[X.columns] = scalerX.fit_transform(X[X.columns])

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)
accuracies = {}

from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier()
ada.fit(X_train,y_train)
ada_pre=ada.predict(X_test)
acc_ada = accuracy_score(y_test,ada_pre)*100
accuracies['AdaBoostClassifier'] = acc_ada
print("Test Accuracy {:.2f}%\n".format(acc_ada))


#'Logistic Regression'
print("Logistic Regression:")
model_Log= LogisticRegression(random_state=1)
model_Log.fit(X_train,y_train)
Y_pred= model_Log.predict(X_test)
model_Log_accuracy=round(accuracy_score(y_test,Y_pred), 4)*100 # Accuracy
accuracies['Logistic Regression'] = model_Log_accuracy
print("Test Accuracy {:.2f}%\n".format(model_Log_accuracy))

#KNeighbors
print("KNeighbors:")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

print("{} NN Score: {:.2f}%\n".format(2, knn.score(X_test, y_test)*100))


#'Naive Bayes'
print("Naive Bayes:")
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

nb_acc = nb.score(X_test,y_test)*100
accuracies['Naive Bayes'] = nb_acc
print("Accuracy of Naive Bayes: {:.2f}%\n".format(nb_acc))


#'Decision Tree'
print("Decision Tree:")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

dtc_acc = dtc.score(X_test, y_test)*100
accuracies['Decision Tree'] = dtc_acc
print("Decision Tree Test Accuracy {:.2f}%\n".format(dtc_acc))

###
model_svm=SVC(kernel="rbf")
model_svm.fit(X_train,y_train)
Y_pred=model_svm.predict(X_test)
model_svm_accuracy=round(accuracy_score(y_test,Y_pred), 4)*100
accuracies['Support Vector Machine'] = model_Log_accuracy
print("Support Vector Machine Test Accuracy {:.2f}%\n".format(model_svm_accuracy))

key = ['LogisticRegression','KNeighborsClassifier','DecisionTreeClassifier','AdaBoostClassifier']
value = [LogisticRegression(random_state=9), KNeighborsClassifier(), DecisionTreeClassifier(),AdaBoostClassifier()]
models = dict(zip(key,value))

predicted =[]

for name,algo in models.items():
    model=algo
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test, predict)
    predicted.append(acc)
    print(name,acc)


