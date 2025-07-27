# Some commands for building and running a Docker container for inference locally

docker build . -t tia-pegasus -f docker_task_1/dockerfile
docker run -it --gpus all --network none tia-pegasus

tar -czvf algorithmmodel.tar.gz -C weights/ .