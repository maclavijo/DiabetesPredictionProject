### Diabetes Prediction Project

This project is based on the [Kaggle dateset](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook). The dataset is a smaller and cleaner version of the dataset published by the CDC - Behavioral Risk Factor Surveillance System in 2015 which can be found [here](https://www.cdc.gov/brfss/annual_data/annual_2015.html).

The target variable is a binary value that represets whether the person has diabetes 1 or not 0 and the features are numerical and categorical. This project compares the predicted probability of 5different ML models Decision Trees, Logistic Regression, Random Forest, XGBoost and AdaBoostClassifier. It will also provide the changein probability (delta) when the parameters vary w.r.t the previous inputed data.

Note: If you dont know you BMI you can calcuate it from [here](https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html).
Disclaimer: This project is not intended to provide or replace any health or medical advise provided by health professionals. Its for self-educational purposes only.
You can follow me on [Github](https://github.com/maclavijo) and find this project [here](https://github.com/maclavijo/Projects/tree/main/Diabetes_Prediction) and give it a ⭐ if you may 💙.

### File structure
|   diabetes.ipynb<br>
|   diabetes_app.py<br>
|   Pipfile<br>
|   Pipfile.lock<br>
|   Predict.py<br>
|   Previous.txt<br>
|   README.md<br>
|   train.py<br>
|   utils.py<br>
|<br>
+---Datasets<br>
|       diabetes_dataset.csv<br>
|<br>
+---models<br>
|       DecisionTreeClassifier.bin<br>
|       LogisticRegression.bin<br>
|       RandomForestClassifier.bin<br>
|       XGBClassifier.bin<br>

### Project was deployed to streamlit cloud
It can be found [here](https://maclavijo-projects-diabetes-predictiondiabetes-app-z713qp.streamlit.app/)
