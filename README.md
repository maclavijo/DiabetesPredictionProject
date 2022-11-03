### Diabetes Prediction Project

This project is based on the [Kaggle dateset](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook). The dataset is a smaller and cleaner version of the dataset published by the CDC - Behavioral Risk Factor Surveillance System in 2015 which can be found [here](https://www.cdc.gov/brfss/annual_data/annual_2015.html).

The target variable is a binary value that represets whether the person has diabetes 1 or not 0 and the features are numerical and categorical. This project compares the predicted probability of 5different ML models Decision Trees, Logistic Regression, Random Forest, XGBoost and AdaBoostClassifier. It will also provide the changein probability (delta) when the parameters vary w.r.t the previous inputed data.

Note: If you dont know you BMI you can calcuate it from [here](https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html).
Disclaimer: This project is not intended to provide or replace any health or medical advise provided by health professionals. Its for self-educational purposes only.
You can follow me on [Github](https://github.com/maclavijo) and find this project [here](https://github.com/maclavijo/Projects/tree/main/Diabetes_Prediction) and give it a ⭐ if you may 💙.

### File structure
```
├── ...
├── diabetes.ipynb
├── DiabetesPredictionProject
├── Pipfile
├── Pipfile.lock
├── Predict.py
├── Previous.tx
├── README.md
├── train.py
├── utils.py
├── Datasets                    # datasets
│   ├── diabetes_dataset.csv
│   └── ...
├── models
│   ├── DecisionTreeClassifier.bin
│   ├──ogisticRegression.bin
│   ├──andomForestClassifier.bin
│   ├──GBClassifier.bin
│   └── ...
```
### Project was deployed to streamlit cloud
It can be found and run from [here](https://maclavijo-diabetespredic-diabetes-predictiondiabetes-app-9wqx5h.streamlit.app/)

### To run it locally
1. From your console do the following:
- Run command: pipenv install
- Run command: streamlit run DiabetesPredictionProject
- You can now view your Streamlit app in your browser. Local URL[url](http://localhost:8501) or Network URL[url](http://10.97.0.6:8501)
2. Using docker.
- Run command: docker pull supermac789/diabetes_app_streamlit:latest
. Access the app here [http://localhost:8501](http://localhost:8501)

