FROM pytorch/pytorch:latest

RUN mkdir -p /usr/pl-nlp

WORKDIR /usr/pl-nlp

COPY . .

RUN pip install -r requirements.txt

#CMD ["python3", "src/data/make_dataset.py"]

CMD ["python3", "src/models/train_model.py"]