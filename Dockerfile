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
ARG AWS_DEFAULT_REGION

# Set environment variables
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

EXPOSE 5000
CMD ["python", "app.py"]