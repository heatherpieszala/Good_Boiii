import paho.mqtt.client as mqtt
import sys

LOCAL_MQTT_HOST = "mosquitto-service"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = 'dogdetect/images'
REMOTE_MQTT_TOPIC = "dogcapture/images"
REMOTE_MQTT_HOST = "54.82.52.69"
REMOTE_MQTT_PORT = 30355

def on_connect_local(local_client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    local_client.subscribe(MQTT_TOPIC)

def on_connect_remote(remote_client, userdata, flags, rc):
    print("connected to remote broker with rc: " + str(rc))
    #remote_client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print("msg received", str(msg.payload.decode("utf-8")))
    msg = msg.payload
    remote_client.publish(REMOTE_MQTT_TOPIC,payload=msg, qos=1, retain = False)
    print('re-published to cloud!')

#client object for local and remote
print("Local Client")
local_client = mqtt.Client()
local_client.on_connect = on_connect_local
local_client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_client.on_message = on_message
print("Remote Client")
remote_client = mqtt.Client()
remote_client.on_connect = on_connect_remote
remote_client.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

#local_client.loop_start()
remote_client.loop_start()

#loop
#remote_client.loop_forever()
local_client.loop_forever()
