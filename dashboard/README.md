## Docker Compose
runs all necessary containers and networks them properly.  
need to build all the images prior to deployment. (use the given bash scripts for this)

| Script Name           | Description                                                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------------------------|
| build-mqtt.sh         | builds the mqtt docker image. copies the config file given in the mosquitto directory into the docker image. |
| build-nodered.sh      | builds the nodered docker image. copies the given flows.json file placed inside the node_red_data directory. |
| build-predictor.sh    | builds the docker image for the predictor services with python 3.8 as the base docker image.                 |
| build-docker-setup.sh | creates necessary docker volumes and networks. should be executed prior to running docker-compose.           |

after building the images, volumes, and networks required, simply run the following command to run the docker containers.
```shell
docker-compose up -d
```

## Port Mappings
container ports have been mapped as specified below.

| Server/Service | Container Port Exposed | Host Port Mapped |
|----------------|------------------------|------------------|
| MQTT           | 1883                   | 5050             |
| NODE-RED       | 1880                   | 5051             |
| PY-PREDICTOR   | 5052                   | 5052             |


## Testing MQTT Server
- First install MQTT utilities from [here](https://mosquitto.org/download/). version `2.0.14` is preferred. this is required to run mqtt commands in the terminal.
- After installation is done, open up two terminals and run the following commands in order to `subscribe` and `publish` accordingly.  

`LOCALHOST`
```shell
# To subscribe
mosquitto_sub -h localhost -p 5050 -t my-mqtt-topic
```

```shell
# To publish
mosquitto_pub -h localhost -p 5050 -t my-mqtt-topic -m "hello world!"
```

`HIVE MQTT`
```shell
# To subscribe
mosquitto_sub -h broker.hivemq.com -p 1883 -t IOTBDA@SLIIT202219/temp
```

```shell
# To publish
mosquitto_pub -h broker.hivemq.com -p 1883 -t IOTBDA@SLIIT202219/temp -m "{  \"temp\":12,  \"humidity\":23,  \"timestamp\":\"2022-03-01 00:00:00\" }"
```

- the published message should appear in the subscribed terminal if the containers have been correctly configured and deployed.

## Connecting to Node-Red
to visit node-red server, simply open up a browser window and visit `http://localhost:5051`, assuming the default docker-compose ports specified above are being used.
