def InputData():
    patients = {
        'Patient # 1': {
            'highbp': 1,
            'highchol': 1,
            'cholcheck': 0,
            'bmi': 27,
            'smoker': 0,
            'stroke': 1,
            'heartdiseaseorattack': 1,
            'physactivity': 0,
            'hvyalcoholconsump': 1,
            'genhlth': 2,
            'menthlth': 0,
            'physhlth': 0,
            'diffwalk': 1,
            'sex': 0,
            'age': 62,
            'education': 5,
            'income': 7
                    },

        'Patient # 2': {
            'highbp': 0,
            'highchol': 0,
            'cholcheck': 0,
            'bmi': 22,
            'smoker': 1,
            'stroke': 1,
            'heartdiseaseorattack': 0,
            'physactivity': 1,
            'hvyalcoholconsump': 1,
            'genhlth': 3,
            'menthlth': 0,
            'physhlth': 10,
            'diffwalk': 1,
            'sex': 0,
            'age': 52,
            'education': 3,
            'income': 0
            },

        'Myself': {
            'highbp': 1,
            'highchol': 1,
            'cholcheck': 1,
            'bmi': 10,
            'smoker': 1,
            'stroke': 1,
            'heartdiseaseorattack': 1,
            'physactivity': 1,
            'hvyalcoholconsump': 1,
            'genhlth':  0,
            'menthlth': 0,
            'physhlth': 0,
            'diffwalk': 1,
            'sex': 'Male',
            'age': 20,
            'education': 0,
            'income': 0
            }
            }
    return patients


def Description():
    description =   f''\
                f'This project is based on the Kaggle dateset https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook. '\
                f'The dataset is a smaller and cleaner version of the dataset published by the CDC - Behavioral Risk Factor Surveillance System in 2015 which '\
                f'can be found here https://www.cdc.gov/brfss/annual_data/annual_2015.html. <br><br>'\
                f'The target variable is a binary value that represets whether the '\
                f'person has diabetes {1} or not {0} and the features are numerical and categorical. This project compares the predicted probability of 5' \
                f'different ML models Decision Trees, Logistic Regression, Random Forest, XGBoost and AdaBoostClassifier. It will also provide the change' \
                f'in probability (delta) when the parameters vary w.r.t the previous inputed data.<br><br>' \
                f' <strong>Note: If you don''t know you BMI you can calcuate it from here: https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html <br>'\
                f'Disclaimer: This project is not intended to provide or replace any health or medical advise provided by health professionals. '\
                f'It''s for self-educational purposes only. <br>' \
                f'You can follow me on Github https://github.com/maclavijo and find this project here https://github.com/maclavijo/Projects/tree/main/Diabetes_Prediction.</strong>'

    return description