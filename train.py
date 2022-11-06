import os
import time
import pickle
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, f1_score, roc_curve, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# Variables

randomState = 42
url='Datasets/diabetes_dataset.csv'
# Inner cross-validation(for Hyperparameter tuning)
innerCV = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)
# Outer cross-validation(for testing the tunned model)
outerCV = StratifiedKFold(n_splits=3, shuffle=True, random_state=randomState)
scoring = {"auc": "roc_auc", "f1_weighted": "f1_weighted"}

# Helper functions

def getData(url = url):

    url = url
    df = pd.read_csv(url)
    return df

def cleanData(url = url):

    #Drop duplicates
    print('Dropping duplicates...\n')
    df.drop_duplicates(inplace=True)
    time.sleep(1)

    #Change all column names to lower case
    print('Converting to lower case columns and data...')
    df.columns = df.columns.str.replace('Diabetes_binary','diabetes').str.lower()

    #This next for loop doesn't get executed because there are no "object" type columns
    for col in df.select_dtypes(object).columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    time.sleep(1)

    return df

def splitData(df):

    target = df.diabetes
    data = df.drop(columns=['diabetes'])
    dfTrainFull, dfTest, yTrainFull, yTest = train_test_split(data, target, test_size=0.2, random_state=randomState)
    dfTrain, dfVal, yTrain, yVal = train_test_split(dfTrainFull, yTrainFull, test_size=0.25, random_state=randomState)

    print(  f'Dataset has been split in: Training set with {len(yTrain)} samples, '
            f'Validation set with {len(yVal)} samples and Test set with {len(yTest)} samples')

    return dfTrainFull, yTrainFull, dfTrain, yTrain, dfVal, yVal, dfTest, yTest

def printHelper(f1Score, auc):
    
    print('\n---------------------------------')
    print(f'Test set weighted f1-score: {f1Score}')
    print(f'Test set auc: {auc}')
    print('---------------------------------\n')

def printResults(results):  
    print('\n-----------------------------------------')
    for i,j in results.items():
        print('{:<20}:  {:<6}'.format(i, " ± ".join([str(x) for x in j])))
    print('-----------------------------------------')

    
def getMeasures(model):

    yTestpredProb = model.predict_proba(dfTest)[:,1]
    yTestpred = model.predict(dfTest)
    auc = round(roc_auc_score(yTest, yTestpredProb),3)
    f1Score = round(f1_score(yTest, yTestpred, average='weighted'),3)
    modelName = type(model.named_steps.classifier).__name__

    printHelper(f1Score, auc)

    print(classification_report(yTest, yTestpred))

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    fpr, tpr, _ = roc_curve(yTest.values, yTestpredProb)
    roc_display1 = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1, name=f'ROC_AUC {modelName}', )
    ax1.text(0.4,0.5,f'AUC = {auc}', size=14, fontweight='semibold', )
    ax1.text(0.4,0.4,f'Weighted F1  = {f1Score}', size=14, fontweight='semibold')
    ax1.legend(loc=4, prop={'size': 20})

    print()

    fig2, ax2 = plt.subplots(1,2, figsize=(16, 6))
    ax2[0].grid(False)
    ax2[1].grid(False)

    cm = confusion_matrix(yTest, yTestpred)
    cmprob = confusion_matrix(yTest, yTestpred, normalize='true')
    cm_display1 = ConfusionMatrixDisplay(cm, display_labels=['No-Diabetes', 'Diabetes'])
    cm_display2 = ConfusionMatrixDisplay(cmprob, display_labels=['No-Diabetes', 'Diabetes'])

    cm_display1.plot(ax=ax2[0])
    cm_display1.ax_.set_title("Confusion Matrix", size=16)
    cm_display2.plot(ax=ax2[1])
    cm_display2.ax_.set_title("Narmalized Confusion Matrix", size=16)
    
    rtn = {}
    rtn['test_roc_auc'] = [auc]
    rtn['test_f1-score'] = [f1Score]
    
    return rtn #[auc], [f1Score]#, yTestpred, yTestpredProb

def getResults(model, params):
    
    baseParams = ["mean_train_auc",
                  "std_train_auc",
                  "rank_test_auc",
                  "mean_train_f1_weighted",
                  "std_train_f1_weighted",
                  "rank_test_f1_weighted"
    ]
    
    allParams = baseParams + params
    
    cv_results = pd.DataFrame(model.cv_results_)

    res = cv_results[allParams]
    
    if 'param_classifier__reg_lambda' not in params:
        print(res.query('rank_test_auc < 30 & rank_test_f1_weighted < 30').sort_values(by=["rank_test_auc", "rank_test_f1_weighted"]))
    else:
        print(res.sort_values(by=["rank_test_auc", "rank_test_f1_weighted"]).head(20))

def getBestModelResults(model):
    
    cv_results = cross_validate(model,
                                dfTrainFull,
                                yTrainFull,
                                cv=outerCV,
                                scoring=['f1_weighted','roc_auc'],
                                n_jobs=-1,
                                return_train_score=True,
                                return_estimator=True,
    )

    cv_results = pd.DataFrame(cv_results)
    cv_test_scores = cv_results[['test_f1_weighted', 'train_f1_weighted', 'test_roc_auc', 'train_roc_auc']]
    cv_test_scores.columns = ['val_f1_weighted', 'val_roc_auc','train_f1_weighted', 'train_roc_auc']
    
    print("Scores after hyperparameters tuning:\n")
    
    res = cv_test_scores.copy()
    res.loc['mean'] = res.mean().round(4)
    res.loc['std'] = res.std().round(4)
    
    rtn = {}
    for col in res:
        print('{:<20}:  {:<6} +/- {:<6}'.format(col, res.loc["mean"][col], res.loc["std"][col]))
        #print(f'{col}: {res.loc["mean"][col]} +/- {res.loc["std"][col]}')
        rtn[col] = [res.loc["mean"][col], res.loc["std"][col]]
    
    return rtn

#### Load binary unbalanced data
df = getData()

### EDA
##### Preparing and cleaning data
df = cleanData(df)

# Checking feature and target values.
diabetes0 = df[df.diabetes==0]
diabetes1 = df[df.diabetes==1]
columns = df.columns
idxs = list(product(range(0,11), range(0,2)))

fig, axs = plt.subplots(11,2, figsize=(12, 30))
for col,ax in zip(columns, axs.ravel()):
    ax.hist(diabetes0[col], stacked=True, label='No Diabetes')
    ax.hist(diabetes1[col], stacked=True, label='Diabetes')
    ax.legend(prop={'size': 10})
    ax.set_title(col)

fig.tight_layout()
plt.show()  


#### Checking Correlations

dfMatrix = df.drop(columns='diabetes')
corr_matrix = dfMatrix.corrwith(df.diabetes).abs()
plt.figure(figsize=(14,5))
_ = corr_matrix.plot(kind='bar', grid=True)
_ = plt.show()    

print(corr_matrix.sort_values(ascending=False))

### Dropping parameters with lower correlation
df.drop(columns=['nodocbccost', 'fruits', 'anyhealthcare', 'veggies'], inplace=True)

#### The target value is heavily imbalanced.  No Diabetes- 194377, Diabetes- 35097.

_= df.hist(figsize=(16,16))

dfTrainFull, yTrainFull, dfTrain, yTrain, dfVal, yVal, dfTest, yTest = splitData(df)

# Preprocessing
# This part will be used for the Logistic Regression classifier only

categoricalCols = ['highbp', 'highchol', 'cholcheck','smoker','stroke',
                   'heartdiseaseorattack', 'physactivity', 'hvyalcoholconsump',
                   'genhlth','diffwalk', 'sex', 'education', 'income']

numericalCols = ['bmi', 'menthlth', 'physhlth']

# creating preprocesors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
catPreprocessor = OneHotEncoder(handle_unknown="ignore")
numPreprocessor = StandardScaler()

# Transforming the data
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('one-hot-encoder', catPreprocessor, categoricalCols)],remainder="passthrough")

##################
### ML Models
##################

#### DecisionTreeClassifier -  Will use nested CrossValidation

print('DecisionTree Classifier')

model_DT = Pipeline([("classifier", DecisionTreeClassifier(class_weight='balanced', random_state = randomState))])

# Gridsearch params
param_grid = {
    'classifier__max_depth': (1,3,5,7,10),
    'classifier__max_leaf_nodes': (1, 5,10,15,20),
    'classifier__max_features': (1,3,5,7,10,15)
}

# Gridsearch
model_grid_search_DT = GridSearchCV(model_DT,
                                    param_grid=param_grid,
                                    scoring=scoring,
                                    n_jobs=-1,
                                    cv=innerCV,
                                    return_train_score=True,
                                    refit=False)

_ = model_grid_search_DT.fit(dfTrainFull, yTrainFull)

paramsDT = ["param_classifier__max_depth",
            'param_classifier__max_leaf_nodes',
            "param_classifier__max_features"]

getResults(model_grid_search_DT, paramsDT)

# # Selecting best parameters
# We will choose max_depth=7, max_leaf_nodes=20 and max_features=15. Reaching a compromise between F1 Score and AUC.

bestParamsDT = ['max_depth=7', 'max_leaf_nodes=20', 'max_features=5']

modelDT = Pipeline([("classifier", DecisionTreeClassifier(class_weight='balanced',
                                                        random_state = randomState,
                                                        max_depth=7,
                                                        max_leaf_nodes=20,
                                                        max_features=15))])

_ = modelDT.fit(dfTrainFull, yTrainFull)

# Outer cross-validation(for testing the tunned model)

results = {}
results = getBestModelResults(modelDT)

# # Curves and error measures
results.update(getMeasures(modelDT))

#  DecisionTreeClassifier results:
printResults(results)


#### Logistic Regression
print('Logistic Regression Classifier')
model_LR = Pipeline([("processor", preprocessor),
                  ("classifier", LogisticRegression(max_iter=1000,
                                                    class_weight='balanced',
                                                    random_state=randomState))])

param_grid = {
    'classifier__C': (1e-3, 1e-2, 0.1, 1, 5, 10, 20),
}
scoring = {"auc": "roc_auc", "f1_weighted": "f1_weighted"}

model_grid_search_LR = GridSearchCV(model_LR,
                                 param_grid=param_grid,
                                 scoring=scoring,
                                 n_jobs=-1,
                                 cv=innerCV,
                                 return_train_score=True,
                                 refit=False)
_ = model_grid_search_LR.fit(dfTrainFull, yTrainFull)

paramsLR = ["param_classifier__C"]
getResults(model_grid_search_LR, paramsLR)

##### Selecting best parameters
#Choosing C=0.01, in this case it's the beast AUC and 2nd best F1 Score
modelLR = Pipeline([("classifier", LogisticRegression(max_iter=1000,
                                                    C=0.01,
                                                    class_weight='balanced',
                                                    random_state=randomState))])

_ = modelLR.fit(dfTrainFull, yTrainFull)

# Outer cross-validation(for testing the tunned model)
results = {}
results = getBestModelResults(modelLR)

##### Curves and error measures
results.update(getMeasures(modelLR))

#### Logistic Regression results:
printResults(results)


#### Random Forest
print('Random Forest Classifier')

model_RF = Pipeline([("classifier", RandomForestClassifier(n_estimators=10,
                                                           class_weight='balanced',
                                                           random_state=randomState))])

param_grid = {
    'classifier__max_depth': (5,10,15,20,25),
    'classifier__max_leaf_nodes': (5,10,15,20,30),
    'classifier__max_features': (3,5,7,10)
}

scoring = {"auc": "roc_auc", "f1_weighted": "f1_weighted"}

model_grid_search_RF = GridSearchCV(model_RF,
                                 param_grid=param_grid,
                                 scoring=scoring,
                                 n_jobs=-1,
                                 cv=innerCV,
                                 return_train_score=True,
                                 refit=False)
_ = model_grid_search_RF.fit(dfTrainFull, yTrainFull)

paramsRF = ["param_classifier__max_depth",
            'param_classifier__max_leaf_nodes',
            "param_classifier__max_features"]

getResults(model_grid_search_RF, paramsRF)

# # Selecting best parameters
# Choosing max_depth = 10, max_leaf_nodes=30 and max_features=5.
modelRF = Pipeline([("classifier", RandomForestClassifier(n_estimators=10,
                                                          max_depth = 10,
                                                          max_leaf_nodes=30,
                                                          max_features=5,
                                                          class_weight='balanced',
                                                          random_state=randomState))])

_ = modelRF.fit(dfTrainFull, yTrainFull)

# Outer cross-validation(for testing the tuned model)
results = {}
results = getBestModelResults(modelRF)

##### Curves and error measures
results.update(getMeasures(modelRF))


#### Random Forest results:
printResults(results)


#### XGBoost Classifier
print('XGBoost Classifier')

imbalanceRatio = (yTrainFull==0).sum() / (yTrainFull==1).sum()
imbalanceRatio = round(imbalanceRatio, 2)
imbalanceRatio
innerCV = StratifiedKFold(n_splits=3, shuffle=True, random_state=randomState)

model_XGB = Pipeline([("classifier", XGBClassifier(n_estimators=10,
                                                   random_state=randomState, 
                                                   tree_method='gpu_hist',
                                                   scale_pos_weight=imbalanceRatio)
                      )])

param_grid = {
    'classifier__max_depth' : (2, 5, 8, 10),
    'classifier__learning_rate' : (0.01, 0.1, 0.5, 0.8),
    'classifier__min_child_weight' : (1,10,20),
    'classifier__reg_lambda' : (1, 3, 5, 8),
}

scoring = {"auc": "roc_auc", "f1_weighted": "f1_weighted"}

model_grid_search_XGB = GridSearchCV(model_XGB,
                                 param_grid=param_grid,
                                 scoring=scoring,
                                 n_jobs=-1,
                                 cv=innerCV,
                                 return_train_score=True,
                                 verbose=0,                                 
                                 refit=False)

_ = model_grid_search_XGB.fit(dfTrain, yTrain)

paramsXGB = ["param_classifier__max_depth",
             'param_classifier__learning_rate',
             "param_classifier__min_child_weight",
             "param_classifier__reg_lambda"]

getResults(model_grid_search_XGB, paramsXGB)

# # Selecting best parameters
# Choosing max_depth = 8, learning_rate = 0.5, min_child_weight = 20 and reg_lambda = 8,. It's a good compromise between a good AUC and F1 Score.
modelXGB = Pipeline([("classifier", XGBClassifier(n_estimators = 10,
                                                  max_depth = 8,
                                                  learning_rate = 0.5,
                                                  min_child_weight = 20,
                                                  reg_lambda = 8,
                                                  random_state=randomState, 
                                                  #tree_method='gpu_hist',
                                                  scale_pos_weight=imbalanceRatio                                                  
                                                 ))
                    ])

_ = modelXGB.fit(dfTrainFull, yTrainFull)

# Outer cross-validation(for testing the tuned model)
results = {}
results = getBestModelResults(modelXGB)

# # Curves and measures of error
results.update(getMeasures(modelXGB))

#### XGBoost results:
printResults(results)

### Comparing models
models = [modelDT, modelLR, modelRF, modelXGB]

fig, ax = plt.subplots(figsize=(15, 10))
delta = 0

for model in models:

    yTestpredProb_ = model.predict_proba(dfTest)[:,1]
    yTestpred_ = model.predict(dfTest)
    modelName = type(model.named_steps.classifier).__name__
    
    auc = round(roc_auc_score(yTest, yTestpredProb_),3)
    f1Score = round(f1_score(yTest, yTestpred_, average='weighted'),3)
    
    fpr, tpr, _ = roc_curve(yTest.values, yTestpredProb_)
    roc_display1 = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display1.plot(ax=ax, name=modelName)
    roc_display1.ax_.set_title('ROC_AUC and F1 SCORE', size= 16)
    ax.text(0.65, 0.155 - delta, f'auc = {auc}', size=14 )
    ax.text(0.78, 0.155 - delta, f'weighted f1  = {f1Score}', size=14)
    ax.legend(loc='lower center', prop={'size': 14})
    delta += 0.043  


## In this case the best model are XGBoost and Logistc regression.

### Saving Models with Pickle
if not os.path.exists('models'):
    os.mkdir('models')

models = [modelDT, modelLR, modelRF, modelXGB]
for model in models:
    modelName = type(model.named_steps.classifier).__name__
    print(f'Saving pickle file for {modelName}')
    outputFile = f'models/{modelName}.bin'
    with open(outputFile, 'wb') as f:
        pickle.dump(model, f)