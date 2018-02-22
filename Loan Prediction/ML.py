import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from xgboost import XGBClassifier


def classification_model(model, data, predictors, outcome):

    model.fit(data[predictors], data[outcome])

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(data.shape[0], n_folds=5)
    error = []

    for train, test in kf:
        train_predictors = (data[predictors].iloc[train, :])

        train_target = data[outcome].iloc[train]

        model.fit(train_predictors, train_target)

        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))


    print("Cross-Validation Score : %s" % "{0: .3%}".format(np.mean(error)))

    #Fit the model again so that it can be referred outside the function:
    model.fit(data[predictors], data[outcome])


train = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data//train_final.csv")
test = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data//test.csv")

# train.drop(["Unnamed: 0","index"], axis=1, inplace=True)
# test.drop(["Unnamed: 0","index"], axis=1, inplace=True)

## Convert categorical to numeric variables
## LabelEncoder --> Ordinal variables
## One_hot_encoding(pd.dummy.. ) --> binary variable(high dimensional)
cat_var = ["Education", "Gender", "Married", "Property_Area", "Self_Employed", "Dependents"]

le = LabelEncoder()

for i in cat_var:
    train[i] = le.fit_transform(train[i])


## Logistic Regression
print("Logistic Regression:")
outcome_var = "Loan_Status"
model = LogisticRegression()

predictor_var = ["Credit_History"]
classification_model(model, train, predictor_var, outcome_var)

predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, train, predictor_var, outcome_var)

## Decision Tree
print("\nDecision Tree:")
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, train, predictor_var,outcome_var)

predictor_var = ['Credit_History','Loan_Amount_Term']
classification_model(model, train, predictor_var,outcome_var)


## Random Forest

print("\nRandom Forest:")
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Credit_History', 'Dependents', 'Education', 'Gender', 'LoanAmount',
       'Loan_Amount_Term', 'Married', 'Property_Area',
       'Self_Employed', 'total_income']
classification_model(model, train,predictor_var,outcome_var)

"""
Above random forst gives train accuracy of 100% and CV accuracy as 80% which is ultimate case of overfitting
This can be resolved in 2 ways:
    1. Reducing No. of predictors
    2. Tuning the model parameters
Thus we calculate feature imp and only use top 5 features
"""

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print("\nFeature Importances:\n",featimp)


## Using top 5 imp features
print("\nRandom Forest with top 5 imp features")
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['Credit_History', 'total_income', 'LoanAmount', 'Dependents', 'Property_Area']
classification_model(model, train, predictor_var, outcome_var)


## XGBoost Classifier
print("\nXGBoost Classifier:")

model = XGBClassifier()
predictor_var = ['Credit_History', 'Dependents', 'Education', 'Gender', 'LoanAmount',
       'Loan_Amount_Term', 'Married', 'Property_Area',
       'Self_Employed', 'total_income']
classification_model(model, train, predictor_var, outcome_var)



