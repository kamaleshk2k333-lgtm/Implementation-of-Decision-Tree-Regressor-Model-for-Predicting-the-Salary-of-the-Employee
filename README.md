# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import required libraries like pandas and sklearn.

3.Read the Salary.csv dataset.

4.Check dataset using head(), info(), and isnull().

5.Convert the Position column into numerical values using LabelEncoder.

6.Select Position and Level as input features (X).

7.Select Salary as output variable (Y).

8.Split the dataset into training and testing data.

9.Train the Decision Tree Regressor model using training data.

10.Predict salary using test data and calculate R² score to check performance. 
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: kamaleshkumar k
RegisterNumber: 25012000
*/
import pandas as pd
data = pd.read_csv("Salary.csv")

data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

data.head()

x = data[["Position", "Level"]]
x.head()

y = data["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
y_pred

from sklearn import metrics
r2 = metrics.r2_score(y_test, y_pred)
r2

```

## Output:
<img width="728" height="257" alt="555183407-42ed7234-5150-4ec3-b695-7655f24d3429" src="https://github.com/user-attachments/assets/6445f9c6-8993-44b9-a715-9c7cb2867c7c" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
