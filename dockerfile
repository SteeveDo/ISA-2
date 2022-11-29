FROM python:3.9

WORKDIR /code/app

COPY ./requirements.txt /code/requirements.txt

COPY ./fr_core_news_sm-3.4.0.tar.gz /code/app/fr_core_news_sm-3.4.0.tar.gz

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app/

EXPOSE 7000

CMD ["python", "main.py"]