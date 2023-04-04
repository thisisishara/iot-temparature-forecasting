docker volume create --driver local \
    --opt type=none \
    --opt device=./node_red_data \
    --opt o=bind node_red_configs

docker volume create --driver local \
    --opt type=none \
    --opt device=./mosquitto \
    --opt o=bind mosquitto_configs

docker volume create --driver local \
    --opt type=none \
    --opt device=./pypredict_data \
    --opt o=bind pypredict_configs

docker network create sliit_raspberry_pi_network

#docker volume create --driver local --opt type=none --opt device=./node_red_data --opt o=bind node_red_configs
#docker volume create --driver local --opt type=none --opt device=./mosquitto --opt o=bind mosquitto_configs
#docker volume create --driver local --opt type=none --opt device=./pypredict_data --opt o=bind pypredict_configs
#docker network create sliit_raspberry_pi_network
