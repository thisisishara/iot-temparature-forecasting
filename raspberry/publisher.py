import time
import paho.mqtt.client as paho
from time import sleep
from datetime import datetime
import logging
import db_read as DB

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_TEMP = "IOTBDA@SLIIT202219/temp"
MQTT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('MQTTPublisher\t%(asctime)s\t%(levelname)s\t%(name)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


# on connect callback
def on_connect(client_name, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)


# on publish callback
def on_publish(client_name, userdata, mid, properties=None):
    print("Published temperature: " + str(mid))


def trigger_publisher():
    client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
    client.connect(MQTT_BROKER, MQTT_PORT)  # connect to HiveMQ

    # registering callbacks
    client.on_connect = on_connect
    client.on_publish = on_publish

    # publishing temperature
    client.loop_start()
    while True:
        try:
            temperature = DB.get_db_temperature_c()  # Reading the temperature from the sensor
            humidity = DB.get_db_humidity()
        except KeyboardInterrupt:
            break
        except:
            print("skipped reading")
            continue

        print(time.time(), '\t', temperature, humidity)  # TODO: Use a logger instead of print statements

        # Use the following JSON format when streaming temperature
        timestamp = datetime.now().strftime(MQTT_TIMESTAMP_FORMAT)
        temp_string = f"{{  \"temp\":{temperature},  \"humidity\":{humidity},  \"timestamp\":\"{timestamp}\" }}"
        client.publish(MQTT_TOPIC_TEMP, payload=temp_string, qos=1)
        sleep(1)


if __name__ == "__main__":
    trigger_publisher()

