import json
import numpy as np
import paho.mqtt.client as mqtt
import onnxruntime as ort
import hashlib

# MQTT & Model Configuration
BROKER = "mosquittobroker"
PORT = 1883
SUBSCRIBE_TOPIC = "input/measurement"
PUBLISH_TOPIC = "output/predictk2001"
MODEL_PATH = "xgboostmodel_mqtt_prediction.onnx"

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Utility: Convert string to float hash
def hash_string(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000 / 1000.0

# Updated: Parse flat input JSON & build input vector
def preprocess_input(msg_json):
    instrument = hash_string(msg_json["Instrument"])
    measurement_attr = float(msg_json["Measurement_attr"])
    measurement_value = float(msg_json["Measurement_value"])
    serial_no = hash_string(msg_json["Serial_no"])
    device_id = hash_string(msg_json["deviceId"])

    # Input vector as per model's expected feature order
    return np.array([[instrument, measurement_attr, measurement_value, serial_no, device_id]], dtype=np.float32)

# Handle MQTT message
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        print(f"Received payload: {payload}")
        input_json = json.loads(payload)

        features = preprocess_input(input_json)
        prediction = session.run([output_name], {input_name: features})[0][0]

        # Convert NumPy float to native float for JSON
        response = json.dumps({"predicted_K2001": float(prediction)})
        client.publish(PUBLISH_TOPIC, response)
        print(f"Published prediction: {response}")

    except Exception as e:
        print(f"Error processing message: {e}")

# Set up MQTT client
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.subscribe(SUBSCRIBE_TOPIC)

print(f"Listening for inputs on '{SUBSCRIBE_TOPIC}'...")
client.loop_forever()