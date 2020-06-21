FROM python:3.8-slim-buster

RUN mkdir -p /usr/pl-nlp

WORKDIR /usr/pl-nlp

COPY . .

