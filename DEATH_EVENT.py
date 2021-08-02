import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score


#Suppressing all warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('C:/Users/Michael Eliyahu/Downloads/machine learning/final_froject/input/heart_failure_clinical_records_dataset.csv')
dataset.head()

plt.rcParams['figure.figsize']=15,6
sns.set_style("darkgrid")


##show whigth
x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


x = dataset[['ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time','age']]
y = dataset['DEATH_EVENT']

#Spliting data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=65)



#Build Model Linear Regression

model_Log= LogisticRegression(random_state=1)
model_Log.fit(X_train,Y_train)
Y_pred= model_Log.predict(X_test)
model_Log_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

#Build Model  K Nearest Neighbors

model_KNN = KNeighborsClassifier(n_neighbors=15)
model_KNN.fit(X_train,Y_train)
Y_pred = model_KNN.predict(X_test)
model_KNN_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy



#Build Model
model_tree=DecisionTreeClassifier(random_state=10,criterion="gini",max_depth=100)
model_tree.fit(X_train,Y_train)
Y_pred=model_tree.predict(X_test)
model_tree_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy


#Adaboost
print("Adaboost:")
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
model1 = abc.fit(X_train, Y_train)
y_pred = model1.predict(X_test)
Adaboost_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

#'Naive Bayes'
print("Naive Bayes:")
from sklearn.naive_bayes import GaussianNB
model_naive_bayes = GaussianNB()
model_naive_bayes.fit(X_train, Y_train)
y_pred = model_naive_bayes.predict(X_test)
naive_bayes_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

accuracies = {"Logistic regression": model_Log_accuracy,
              "KNN": model_KNN_accuracy,
              "Decision Tree": model_tree_accuracy,
              "Adaboost": Adaboost_accuracy,
              "Naive Bayes:": naive_bayes_accuracy}

acc_list = accuracies.items()
k, v = zip(*acc_list)
temp = pd.DataFrame(index=k, data=v, columns=["Accuracy"])
temp.sort_values(by=["Accuracy"], ascending=False, inplace=True)

# Plot accuracy for different models
plt.figure(figsize=(20, 7))
ACC = sns.barplot(y=temp.index, x=temp["Accuracy"], label="Accuracy", edgecolor="black", linewidth=3, orient="h",
                  palette="twilight_r")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.xlim(80, 98)

ACC.spines['left'].set_linewidth(3)
for w in ['right', 'top', 'bottom']:
    ACC.spines[w].set_visible(False)

# Write text on barplots
k = 0
for ACC in ACC.patches:
    width = ACC.get_width()
    plt.text(width + 0.1, (ACC.get_y() + ACC.get_height() - 0.3), s="{}%".format(temp["Accuracy"][k]),
             fontname='monospace', fontsize=14, color='black')
    k += 1

plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


###PART B



