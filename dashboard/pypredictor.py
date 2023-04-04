import json
import logging
import os.path
from typing import Optional

import pandas as pd
from flask import Flask, jsonify, Response, render_template
from datetime import datetime
from paho.mqtt import client as paho
from apscheduler.schedulers.background import BackgroundScheduler

from pyarima import (
    IOTArimaTemperature as ARIMAHandler,
    PredictionStartPoint,
    PredictionsType,
)
from pystream_handler import (
    DataStreamHandler
)

VERSION = "0.0.1"

# credentials
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_BASE = "IOTBDA@SLIIT202219"
MQTT_TOPIC_TEMP = "IOTBDA@SLIIT202219/temp"
MQTT_STREAM_DIR = "./pypredict_data/temp_stream"
MQTT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('PyPredictor\t%(asctime)s\t%(levelname)s\t%(name)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

app = Flask(__name__)

# global temp var
current_streaming_temp: str = "Current Temperature from Raspberry Pi"
current_streaming_humidity: str = "Current Humidity from Raspberry Pi"


# scheduler
def get_current_temperature() -> Optional:
    logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\tStarted the job scheduler.")
    client = paho.Client(client_id=None, userdata=None, protocol=paho.MQTTv5)
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT)

    # subscribe to all topics by using the wildcard "#"
    # client.subscribe(MQTT_TOPIC_BASE + "/#", qos=1)
    client.subscribe(MQTT_TOPIC_TEMP, qos=1)
    client.loop_forever()


# mqtt broker
def on_connect(mqtt_client, userdata, flags, rc, properties=None) -> Optional:
    logger.info("CONNECTED: received with code %s." % rc)


def on_subscribe(mqtt_client, userdata, mid, granted_qos, properties=None) -> Optional:
    logger.info("SUBSCRIBED: " + str(mid) + " " + str(granted_qos))


def on_message(mqtt_client, userdata, msg) -> Optional:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("GOT MESSAGE: " + current_datetime + "\t" + msg.topic + " " + str(msg.payload.decode('utf-8')))

    if msg.topic == MQTT_TOPIC_TEMP:
        try:
            payload = json.loads(str(msg.payload.decode('utf-8')))

            if len(payload) != 3:
                logger.error("Incorrect data stream format detected.")
                return

            temperature = float(payload["temp"])
            humidity = float(payload["humidity"])
            timestamp = payload['timestamp']

            # save to csv
            DataStreamHandler.save_stream_to_csv(temp=temperature, timestamp=timestamp)

            # update the global variables to enable flask real-time stream
            global current_streaming_temp
            global current_streaming_humidity
            current_streaming_temp = temperature
            current_streaming_humidity = humidity

        except ValueError:
            logger.error("Could not decode received Raspberry Pi data stream.")
        except Exception as e:
            logger.error(f"Unknown exception occurred while decoding incoming streamed data. More Info: {e}")


with app.app_context():
    # start_scheduler
    scheduler = BackgroundScheduler(daemon=True)
    # scheduler.add_job(get_current_temperature, 'interval', seconds=3)
    scheduler.add_job(get_current_temperature)
    scheduler.start()


@app.route("/")
def dashboard():
    return jsonify({"payload": f"Hello! greetings from py_predictor {VERSION}"})


@app.route("/status")
def status():
    return jsonify(
        [
            {"server": "py_predictor"},
            {"status": "up"}
        ]
    )


@app.route("/train_init")
def train_init():
    model_name = ARIMAHandler.train_initial_model()
    if not model_name:
        return jsonify(
            [
                {"training_status": "failed"},
            ]
        )
    return jsonify(
        [
            {"training_status": "succeeded"},
            {"model_name": model_name}
        ]
    )


@app.route("/update_latest")
def update_latest():
    updated_temp_readings = pd.Series()
    model_name = ARIMAHandler.update_model(new_data=updated_temp_readings)
    if not model_name:
        return jsonify(
            [
                {"updating_status": "failed"},
            ]
        )
    return jsonify(
        [
            {"updating_status": "succeeded"},
            {"model_name": model_name}
        ]
    )


@app.route("/predict_latest", methods=['GET', 'POST'])
def predict_latest():
    predicted_series = ARIMAHandler.get_predictions_from_latest_model(
        pred_type=PredictionsType.JSON,
        mid_point=PredictionStartPoint.MODEL_END,
    )
    if not predicted_series:
        return jsonify(
            [
                {"predictions_status": "failed_to_retrieve"},
            ]
        )
    return jsonify(
        [
            {"predictions_status": "retrieved"},
            {"predicted_series": predicted_series}
        ]
    )


@app.route("/update_and_predict", methods=['GET', 'POST'])
def update_and_predict():
    logger.info("Started updating the ARIMA model.")
    updated_path_ = ARIMAHandler.update_model()
    logger.info(f"Updated the ARIMA model. Updated Model: {updated_path_}")

    predicted_series = ARIMAHandler.get_predictions_from_latest_model(
        pred_type=PredictionsType.JSON,
        mid_point=PredictionStartPoint.UPTODATE,
    )
    if not predicted_series:
        return jsonify(
            [
                {"predictions_status": "failed_to_retrieve"},
            ]
        )
    return jsonify(
        [
            {"predictions_status": "retrieved"},
            {"predicted_series": predicted_series}
        ]
    )


@app.route('/temp_stream')
def temp_stream():
    def generate():
        global current_streaming_temp
        yield f"{datetime.now().strftime(MQTT_TIMESTAMP_FORMAT)}\t:\t{current_streaming_temp}"
    return Response(generate(), mimetype='text')


@app.route('/temp')
def temp():
    return render_template('temp_stream.html')


if __name__ == "__main__":
    # # Uncomment if training an initial ARIMA model prior to starting the server is required
    # # This will be done automatically when the Node-RED starts, thus commented.
    # try:
    #     path_ = ARIMAHandler.train_initial_model()
    #     logger.info(f"Trained an initial ARIMA model. Latest Model Available: {ARIMAHandler.get_latest_model_name()}")
    # except Exception as e:
    #     logger.error("Failed to train the initial ARIMA model. "
    #                  f"There may be errors when trying to generate predictions. More Info: {e}")

    # Starting the flask server
    app.run(host='0.0.0.0', debug=False, port=1881)
