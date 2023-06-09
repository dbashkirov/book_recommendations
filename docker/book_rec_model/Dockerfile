FROM python:3.9

WORKDIR /code

RUN pip install -U pip

COPY ./requirements.txt /code/

COPY ./app.py /code/app/app.py
COPY ./.env /code/app/.env

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8081"]
