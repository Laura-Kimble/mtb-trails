FROM python:3.7

WORKDIR  /DSI/mtb/mtb-trails/web_app


COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python", "./app.py" ]