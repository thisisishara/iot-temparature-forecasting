import time
import paho.mqtt.client as paho
from paho import mqtt

# credentials
mqtt_broker = "broker.hivemq.com"
mqtt_port = 1883

# setting callbacks for different events to see if it works, print the message etc.
def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

# print which topic was subscribed to
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

# print message, useful for checking if it was successful
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload.decode('utf-8')))

#----------------------------------------------------------------------------------------------

mqtt_topic = "DIAN_ze6YppGrxWyV/temperature"

#----------------------------------------------------------------------------------------------

client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
client.on_connect = on_connect
# connect to HiveMQ
client.connect(mqtt_broker, mqtt_port)

# setting callbacks, use separate functions like above for better visibility
client.on_subscribe = on_subscribe
client.on_message = on_message

#----------------------------------------------------------------------------------------------

# subscribe to all topics of testsensortopic by using the wildcard "#"
#client.subscribe("testsensortopic/#", qos=1)
client.subscribe(mqtt_topic, qos=1)

# loop_forever for simplicity, here you need to stop the loop manually
# you can also use loop_start and loop_stop
client.loop_forever()
