FROM python:3.9
WORKDIR /
COPY requirements.txt requirements.txt
COPY . .

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]