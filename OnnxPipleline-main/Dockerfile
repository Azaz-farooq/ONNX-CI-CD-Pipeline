# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MQTT_BROKER="mosquittobroker"
ENV MQTT_PORT=1883
ENV MQTT_KEEPALIVE=60
ENV MQTT_SUBSCRIBE_TOPIC="input/measurement"
ENV MQTT_PUBLISH_TOPIC="output/predictk2001"
ENV ONNX_MODEL_PATH="xgboostmodel_mqtt_prediction.onnx"


# Set working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install core dependencies.
#RUN apt-get update && apt-get install -y libpq-dev build-essential

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run your script
CMD ["python", "onnx-mqtt-marposs.py"]