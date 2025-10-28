FROM python:3.8.12-slim

RUN pip install pipenv


WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py","model_C=0.1.bin","./"]