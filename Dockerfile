FROM python:3.9
WORKDIR /
COPY requirements.txt requirements.txt
COPY . .

RUN apt-get update
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app