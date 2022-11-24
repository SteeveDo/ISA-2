FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


COPY ./requirements.txt /app/requirements.txt

COPY ./fr_core_news_sm-3.4.0.tar.gz /app/fr_core_news_sm-3.4.0.tar.gz

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR /app/app

EXPOSE 7000

COPY ./app /app/app