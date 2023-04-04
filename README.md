Please go under edit and edit this file as needed for your project

# Team Name - 2022_19
# Project Group - 2022_19
### Group Leader - IT19051208 - Mr. Sakalasooriya S.A.H.A. ([akalankasakalasooriya](https://github.com/akalankasakalasooriya))
### Member 2 - IT19069432 - Mr. Dissanayake D.M.I.M. ([thisisishara](https://github.com/thisisishara))
### Member 3 - IT19075754 - Ms. Jayasinghe D.T. ([dinushiTJ](https://github.com/dinushiTJ))
### Member 4 - IT19045986 - Ms. De Silva N. N. M ([Nithya980711](https://github.com/Nithya980711))

#### Brief Description of your Solution
The overall solution consists of a Flask server, MQTT Event Broker, and a Node-RED dashboard in addition to the Raspberry Pi 4, temperature sensor and the gauge. Raspberry Pi streams the collected temperature data from the sensor in real-time while the gauge shows three ranges of temperature, namely, cool (0-20), warm (20-30), and hot (>30). MQTT publisher which resides in the raspberry pi board is responsible for streaming the temperature data to a unique topic and Hive MQ is used as the cloud mosquitto event broker. Then from the Node-RED Dashboard and the Flask server deployed in a remote PC device is reading the temperature readings in real-time using a MQTT Subscriber both from Node-RED Dashboard and the flask server.

Flask server runs a background scheduler which runs the MQTT Subscriber loop and it averages each temperature reading received over a period of month and saves into a CSV file. This CSV file is then utilized by the seasonal ARIMA model to update itself every time the Node-RED dashboard requests for model predictions. If the model is already up-to-date, it provides 12 past temperature values collected with 12 future predictions in a JSON format that is supported by Node-RED dashboard to generate a graph.

The overall solution is integrated with docker, Node-RED and Flask server have been containerized and uploaded to DockerHub where anyone can pull and give a try. The docker-compose file provided with the sorce-code can be run with the simple command, `docker-compose up -d` where it will automatically pull all required images and run the container cluster.

The servers are accessible through the following ports after the docker-compose deployment.
* Flask Server: `http://localhost:5052`
* Flask Real-time temperature values inspecting: `http://localhost:5052/temp`
* Flask Get Predictions from the latest model available: `http://localhost:5052/predict_latest`
* Flast Update existing model and get predictions: `http://localhost:5052/update_and_predict`
* Node-RED Dashboard UI: `http://localhost:5052/ui`
* Node-RED Dashboard Flows: `http://localhost:5052`

Docker Hub Image Links:
* [Flask Server Image](https://hub.docker.com/r/thisisishara/iotbda-predictor-2022-19)
* [Node-RED Dashboard Image](https://hub.docker.com/r/dinushitj/iotbda-node-red-2022-19)
