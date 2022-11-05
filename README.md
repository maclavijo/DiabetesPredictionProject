## Diabetes Prediction Project

### Problem description

This project is based on the [Kaggle dateset](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook). The dataset is a smaller and cleaner version of the dataset published by the CDC - Behavioral Risk Factor Surveillance System in 2015 which can be found [here](https://www.cdc.gov/brfss/annual_data/annual_2015.html).

Our mission will be to predict the probability of someone having diabetes based on different features (which are the responses to the survey) and compare their performance.
The target variable is a binary value that represets whether the person has diabetes 1 or not 0 and the features are numerical and categorical. This project calculates and compares the predicted probability of 5 different ML models: Decision Trees, Logistic Regression, Random Forest, XGBoost and AdaBoostClassifier. It will also provide the changein probability (delta) when the parameters vary w.r.t the previous inputed data, this will allow to see how changeson the different features could affect the predicted probability.

Note: If you dont know you BMI you can calculate it from [here](https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html).
Disclaimer: This project is not intended to give or replace any health or medical advice provided by health proffesionals. Its for educational purposes only.
You can follow me on [Github](https://github.com/maclavijo) and find this project [here](https://github.com/maclavijo/Projects/tree/main/Diabetes_Prediction) and give it a ⭐ if you may 💙.


### File structure
```
├── ...
├── diabetes.ipynb
├── diabetes_app.py
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── Predict.py
├── Previous.txt
├── README.md
├── train.py
├── utils.py
├── Datasets
│   ├── diabetes_dataset.csv
│   └── ...
├── models
│   ├── DecisionTreeClassifier.bin
│   ├── LogisticRegression.bin
│   ├── RandomForestClassifier.bin
│   ├── XGBClassifier.bin
│   └── ...
```

### Dependency and enviroment management

Pipfile and Pipfile.lock files are provided. Copy the content of this folder to your machine. Then from the terminal of your IDE of preference (in the correct work directory) the following:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pipenv install<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pipev shell

Now you will be in the virtual environment and will be able to run the files locally

### To run the project locally from your machine

From your console (in the correct work directory) and after the environment has been created (previous step) run the following command:

streamlit run diabetes_app.py

You can now view your Streamlit app in your browser.
Local URL[http://localhost:8501](http://localhost:8501) or Network URL[http://10.97.0.6:8501](http://10.97.0.6:8501)


### Containerization

Dockerfile has been provided. To create and run the image, from your IDE terminal do the following (within the work directory):

1. First option: Create and run the app yourself.<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Create:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docker build -t diabetes_app_streamlit .<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docker run -p 8501:8501 diabetes_app_streamlit<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can now access the Streamlit app in your web browser: Local URL: [http://localhost:8501](http://localhost:8501) or from URL: [http://0.0.0.0:8501](URL: http://0.0.0.0:8501)<br>

2. Second option: To run it using docker hub repository:<br>

&nbsp;&nbsp;&nbsp;Download image from hub run command:  docker pull supermac789/diabetes_app_streamlit:latest<br>
&nbsp;&nbsp;&nbsp;Run the command from your terminal:<br>
&nbsp;&nbsp;&nbsp;You can now access the Streamlit app in your web browser: Local URL: [http://localhost:8501](http://localhost:8501) or from URL: [http://0.0.0.0:8501](URL: http://0.0.0.0:8501)<br>

### Cloud deployment - Streamlit cloud

The app can be found and run from [https://maclavijo-diabetespredic-diabetes-predictiondiabetes-app-9wqx5h.streamlit.app/](https://maclavijo-diabetespredic-diabetes-predictiondiabetes-app-9wqx5h.streamlit.app/).