import paho.mqtt.client as mqtt
import sys

LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="dogdetect/images"
REMOTE_MQTT_HOST="204.236.165.235"
REMOTE_MQTT_PORT=31378
REMOTE_MQTT_TOPIC = "dogcapture/images"

def on_log(client, userdata, level, buf):
    print("log: ", buf)

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    local_mqttclient.subscribe(LOCAL_MQTT_TOPIC)

def on_connect_remote(client, userdata, flags, rc):        
    print("connected to remote broker with rc: " + str(rc))

def on_message_received(client,userdata,msg):
    print('Image recevied!')

def on_message(client,userdata,msg):
    try:
        #print("message received: ",str(msg.payload.decode("utf-8")))
        # if we wanted to re-publish this message, something like this should work
        msg = msg.payload
        remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=1, retain=False)
        print('message re-published to cloud!')
    except:
        print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
#remote_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local

remote_mqttclient = mqtt.Client()
remote_mqttclient.on_connect = on_connect_remote

local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

local_mqttclient.on_message = on_message

#remote_mqttclient.on_message = on_message
remote_mqttclient.on_log = on_log

# go into a loop
remote_mqttclient.loop_start()
local_mqttclient.loop_forever()
