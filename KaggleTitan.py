#Mauricio Hernández López
#Kaggle Titanic Competition
#Submission based on the code by Shreyas Vedpathak

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

sns.catplot(x="Sibsp", col = 'Survived', data=train, kind = 'count', palette='pastel')
sns.catplot(x="Parch", col = 'Survived', data=train, kind = 'count', palette='pastel')


def is_alone(x):
    if  (x['SibSp'] + x['Parch'])  > 0:
        return 1
    else:
        return 0

train['Is_alone'] = train.apply(is_alone, axis = 1)
test['Is_alone'] = test.apply(is_alone, axis = 1)

g = sns.catplot(x="Is_alone", col = 'Survived', data=train, kind = 'count', palette='deep')


g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

f, axes = plt.subplots(2, 1, figsize = (10, 6))

g1 = sns.distplot(train["Fare"], color="red", label="Skewness : %.2f"%(train["Fare"].skew()), ax=axes[0])
axes[0].title.set_text('Before \'log\' Transformation')
axes[0].legend()

train_fare = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g2 = sns.distplot(train_fare, color="blue", label="Skewness : %.2f"%(train_fare.skew()), ax=axes[1])
axes[1].title.set_text('After \'log\' Transformation')
axes[1].legend()

plt.tight_layout()

sns.catplot(x="Sex", y="Survived", col="Pclass", data=train, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')
sns.catplot(x="Sex", y="Survived", col="Embarked", data=train, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')


train = train.drop(['PassengerId','Name','SibSp','Parch'], axis = 1)
test = test.drop(['Name','SibSp','Parch'], axis = 1)



print("Train columns:", ', '.join(map(str, train.columns))) 
print(train.head())
print("\nTest columns:",  ', '.join(map(str, test.columns)))
print(test.head())

print("TRAIN DATA:")
train.isnull().sum()

print("TEST DATA:")
test.isnull().sum()

train.dtypes

numerical = ['Pclass','Age','Is_alone','Fare']
categorical = ['Sex','Ticket','Cabin','Embarked']

features = numerical + categorical
target = ['Survived']
print('Features:', features, '\nTarget:', target)

plt.figure(figsize=(10,8))
correlation_map = sns.heatmap(train.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")


train_set, valid_set = train_test_split(train, test_size = 0.3, random_state = 0)

numerical_transformer = Pipeline(steps=[ ('iterative', IterativeImputer(max_iter = 10, random_state=0)), 
                                                                    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

accuracy = []
classifiers = ['Linear SVM', 'Radial SVM', 'LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier', 'XGBoostClassifier']
models = [ svm.SVC(kernel='linear'), 
                  svm.SVC(kernel='rbf'), 
                 LogisticRegression(), 
                 RandomForestClassifier(n_estimators=200, random_state=0),
                 AdaBoostClassifier(random_state = 0),
                 xgb.XGBClassifier(n_estimators=100)
                 ]

for mod in models:
    model = mod
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
    pipe.fit(train_set[features], np.ravel(train_set[target]))
    prediction = pipe.predict(valid_set[features])
    accuracy.append(pipe.score(valid_set[features], valid_set[target]))

observations = pd.DataFrame(accuracy, index=classifiers, columns=['Score'])
observations.sort_values(by = 'Score', ascending = False)


rand = RandomForestClassifier(n_estimators=200, random_state=0)
pipe_rand = Pipeline(steps=[('preprocessor', preprocessor),  ('model', rand)])

xgb = xgb.XGBClassifier(n_estimators=100)
pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),  ('model', xgb)])

linear_svm = svm.SVC(kernel='linear', C=0.1,gamma=10, probability=True)
pipe_linear = Pipeline(steps=[('preprocessor', preprocessor),  ('model', linear_svm)])

ensemble_all = VotingClassifier(estimators=[('Random Forest Classifier', pipe_rand),
                                                                         ('Linear_svm', pipe_linear),
                                                                        ('XGB', pipe_xgb)], 
                                                                        voting='soft', weights=[3,2,1])

ensemble_all.fit(train_set[features], np.ravel(train_set[target]))
pred_valid = ensemble_all.predict(valid_set[features])

acc_train = round(ensemble_all.score(train_set[features], train_set[target]) * 100, 2)
acc_valid = round(ensemble_all.score(valid_set[features], valid_set[target]) * 100, 2)
print("Train set Accuracy: ", acc_train, "%\nValidation set Accuracy: ", acc_valid, "%")

print("\nConfusion Matrix:\n", confusion_matrix(valid_set[target], pred_valid))
print("\nClassification Report:\n", classification_report(valid_set[target], pred_valid))
model = RandomForestClassifier(n_estimators=200, random_state = 0)

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

pipe.fit(train_set[features], np.ravel(train_set[target]))

pred_valid = pipe.predict(valid_set[features])
acc_ran_train = round(pipe.score(train_set[features], train_set[target]) * 100, 2)
acc_ran_valid = round(pipe.score(valid_set[features], valid_set[target]) * 100, 2)
print("Train set Accuracy: ", acc_ran_train, "%\nValidation set Accuracy: ", acc_ran_valid, "%")

print("\nConfusion Matrix:\n", confusion_matrix(valid_set[target], pred_valid))
print("\nClassification Report:\n", classification_report(valid_set[target], pred_valid))
pred_test = pipe.predict(test[features])

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred_test})
output.to_csv('prediccionResultados.csv', index=False)
plt.show()