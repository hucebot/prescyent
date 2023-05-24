# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN python3 -m pip install --upgrade pip

WORKDIR /prescyent

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

COPY prescyent prescyent
COPY tests tests
COPY examples examples

CMD ["python3", "-m" , "unittest"]