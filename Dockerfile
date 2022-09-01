FROM python:3.9-slim

WORKDIR .

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "inference.py"]
