# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Jaiyantan S
RegisterNumber:  212224100021
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")

data["left"].value_counts()


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

print("data.head() for Salary:")

data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/53b47ad6-4fa7-44fa-a316-088a39aae0d9)

![image](https://github.com/user-attachments/assets/fa79ffef-ae0a-454e-a529-c93313ec1637)

![image](https://github.com/user-attachments/assets/f230a244-ce97-4b4d-b15e-d9d0125977c1)

![image](https://github.com/user-attachments/assets/51cdcd22-577c-462a-960a-b713f62761a3)

![image](https://github.com/user-attachments/assets/c8c10ee9-b256-44b2-87c1-3b796ba93cde)

![image](https://github.com/user-attachments/assets/2eb03fe7-ff14-44ff-8e1b-46791a252002)

![image](https://github.com/user-attachments/assets/749c00b1-302f-4ed9-ac24-216baf75c812)

![image](https://github.com/user-attachments/assets/4d40ca90-3e02-4ec0-8325-f9c32b3652ca)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
