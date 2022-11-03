FROM python:3.10.8-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["Predict.py", "Previous.txt", "diabetes_app.py", "utils.py", "./"]

COPY models ./models/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "diabetes_app.py","--server.port=8501", "--server.address=0.0.0.0"]