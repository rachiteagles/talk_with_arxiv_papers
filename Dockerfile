FROM python:3.9
WORKDIR /
COPY requirements.txt requirements.txt
COPY . .

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Accept environment variables as build arguments
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set environment variables
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

EXPOSE $PORT

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

