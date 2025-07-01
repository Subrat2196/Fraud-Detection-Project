FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/latest_random_forest_model.pkl /app/models/latest_random_forest_model.pkl
COPY models/power_transformer.pkl /app/models/power_transformer.pkl

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]



