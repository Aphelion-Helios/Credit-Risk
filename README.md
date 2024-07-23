# Credit Risk Analysis 

## Documentation

[Project Link- Credit-Risk-Analysis](https://hammadrkh.wixsite.com/portfolio/projects/credit-risk-analysis)


Risk managers will typically collect data on the loan borrowers. The data will usually be in a tabular format, with each row providing details of the borrower, including their income, total loans outstanding, and a few other metrics.

There will also be a column indicating if the borrower has previously defaulted on a loan. We will use this data to build a model that, given details for any loan described above, will predict the probability that the borrower will default (also known as PD: the probability of default). The provided data will be used to train a function that will estimate the probability of default for a borrower. Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.


- We will produce a function that can take in the properties of a loan and output the expected loss.

- Viable techniques to explore in accordance to this task can range from a simple regression or a decision tree to something more advanced. It would also be effiecient to use multiple methods and provide a comparative analysis.

# Important Statistical Tools: sklearn, imblearn, and SMOTE. 


(A Pharaoh, TensorFlow/Pytorch, and StatsModels/Modeler stack could be viable for this task as well depending on the use case and needed result). 



The first step is to import the necessary libraries to create a linear model to get a better understanding of the metrics we are working with!
## Usage/Examples

```Python

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd

# Read in loan data from a CSV file
df = pd.read_csv('./Loan_Data.csv')

# Define the variable features
features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']

# Calculate the payment_to_income ratio
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
    
# Calculate the debt_to_income ratio
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

clf = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000).fit(df[features], df['default'])
print(clf.coef_, clf.intercept_)

# Use the following code to check yourself
y_pred = clf.predict(df[features])

fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
print((1.0*(abs(df['default']-y_pred)).sum()) / len(df))
print(metrics.auc(fpr, tpr))
 
Output():


[[ 8.18520373 0.54490854 0.01994244 -2.77630853 -0.02418391]] [-0.09162643] 0.0037 0.9925106069101026
```

It never hurts to try other methods! We will now utilize a Stacking Ensemble Method to create a more intelligent nueral network that will decide what tests will be most effective for the data contained in our data set and run them! 


In order to do this we will need to make sure to import all the neccessary libraries to train, test, and split our machine learning algorithim for optimal model selection.

```Python

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
```

# Step 1: Preprocessing the data!


```Python
#Step 1: Data Preprocessing
data = pd.read_csv("./Loan_Data.csv")

# Handle missing values (example: filling missing numerical values with median)
data.fillna(data.median(), inplace=True)

# Encode categorical variables (example: one-hot encoding)
data = pd.get_dummies(data)

# Define numerical features (replace with your actual column names)
numerical_features = ["income", "years_employed", "fico_score"]

# Scale numerical features (example: standard scaling)
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
```

# Step 2: Derived Feature Engineering to perform accurate feature selection!

```Python
# Step 2: Feature Engineering
# Create derived features and perform feature selection
X = data.drop("default", axis=1)
y = data["default"]

# Perform feature selection using SelectKBest and ANOVA F-value
selector = SelectKBest(f_classif, k=10)
selector = SelectKBest(f_classif, k='all')
X_selected = selector.fit_transform(X, y) 
```

# Step 3: Hyperparameter Tuning and splitting the data into training sets to test!

```Python
# Step 3: Model Selection and Hyperparameter Tuning
# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define and train Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Define and train Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Define and train Gradient Boosting model
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Define and train Neural Network model
nn = MLPClassifier()
nn.fit(X_train, y_train)
```

# Step 4: Evaluating our Model!


```Python 

# Step 4: Model Evaluation
# Evaluate the models on the testing data
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
nn_pred = nn.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)
```

# Now lets Print out our evaluation metrics and see how the models performed!


```Python

# Print evaluation metrics for all models
print("Decision Tree:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)

print("\nRandom Forest:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

print("\nGradient Boosting:")
print("Accuracy:", gb_accuracy)
print("Precision:", gb_precision)
print("Recall:", gb_recall)
print("F1 Score:", gb_f1)

print("\nNeural Network:")
print("Accuracy:", nn_accuracy)
print("Precision:", nn_precision)
print("Recall:", nn_recall)
print("F1 Score:", nn_f1)


Output()


Decision Tree: 
Accuracy: 0.995 
Precision: 0.9912790697674418 
Recall: 0.9798850574712644 
F1 Score: 0.985549132947977 


Random Forest: 
Accuracy: 0.994 
Precision: 0.9912280701754386 
Recall: 0.9741379310344828 
F1 Score: 0.9826086956521738 


Gradient Boosting:
Accuracy: 0.996 
Precision: 0.9941860465116279 
Recall: 0.9827586206896551 
F1 Score: 0.9884393063583815 


Neural Network: 
Accuracy: 0.9573
Precision: 0.8638888888888889 
Recall: 1.0 
F1 Score: 0.8785310734463276
```
 


It is important to note here that eventhough our models did get a high level of accuracy, it is still important to restest them before deployment as our data set and the steps we had taken before very may well have influenced the results/statistical significance of the tests through recall bias!!! Considering this model was not adequately fine-tuned for production level standards, we would need to set aside a greater amount of time before we would actually be able to trust it or it's detemination process. 


