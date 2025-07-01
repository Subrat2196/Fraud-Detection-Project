FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/latest_random_forest_model.pkl /app/models/latest_random_forest_model.pkl
COPY models/power_transformer.pkl /app/models/power_transformer.pkl

RUN pip install -r requirements.txt

EXPOSE 5000

# CMD ["python", "app.py"]  -> this is used for local use

# When we are going for eks scaled deployment , we will need below command

CMD ["gunicorn","--bind","0.0.0.0:5000","--timeout","120","app:app"]


