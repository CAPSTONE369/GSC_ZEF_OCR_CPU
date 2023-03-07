FROM python:3.8-buster
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /app
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]