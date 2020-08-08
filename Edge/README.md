# Inference at the EDGE with the NVIDIA Jetson TX2

The following describes the requirements and setup required to replicate the inference model integrated with Amazon Alexa. You will need the following:
* Amazon account 
* Alexa app (can be any Amazon voice activated device like the Echo, an app on a phone/iPad/computer)
* Jetson TX2 (a Raspberry Pi will not be efficient for inferencing)
* Knowledge of Web APIs, Python, Docker and Linux commands

## PART I : Setup Alexa Developer Account

1. Navigate to https://developer.amazon.com/ and create an account (or use an existing Amazon account)
2. Select "amazon alexa" --> "Create Alexa Skills" --> "Console" --> "Create Skill"
3. Enter a Skill name, choose Custom model and Alexa-Hosted (Node.js) and select "Create Skill"
4. Choose the "Hello World Skill" template and select "Continue with template"
5. On the left panel go to "JSON Editor" and replace everything with this. Note the invocation here is "what do you see", you can change this phrase. Keep everything in lowercase if changing. Then save the model (on the top there is a button).
```
{
    "interactionModel": {
        "languageModel": {
            "invocationName": "what do you see",
            "intents": [
                {
                    "name": "YesIntent",
                    "slots": [],
                    "samples": [
                        "yes",
                        "sure"
                    ]
                },
                {
                    "name": "AMAZON.NavigateHomeIntent",
                    "samples": []
                },
                {
                    "name": "AMAZON.StopIntent",
                    "samples": []
                }
            ],
            "types": []
        }
    }
}
```
6. On the left panel, go to the Endpoint tab. Select the HTTPS radio button and fill in the following info for the Default Region.
- The first box will be the endpoint you get from PART V, Step 5
- For the second box, select the option "My development endpoint is a sub-domain of a domain that has a wildcard certificate from a certificate authority"

## PART II : Pre-setup on Jetson TX2
1. Make sure you have redis installed
2. Set up a bridge network. The one used here is called `hw03` and is passed into the docker command like `--network hw03`
3. Identify the IP address of your device. 
4. Download a RTSP app for your Apple or Android phone/device. Note the RTSP IP address. We will be passing this to OpenCV to capture images.
5. Open up 3 terminal windows

## PART III : Redis 

In the first terminal window ...

1. `sudo docker run --name redis1 --network hw03 -d redis`

2. `sudo docker exec -it redis1 sh`

3. Then do `redis-cli` at the prompt

![redis](https://github.com/shwethacn/W251-Final-Project/blob/master/imgs/edge_redis.jpg)

## PART IV : Prediction Model Endpoint

In the second terminal window ...

1. Copy the folder "prediction.zip" from the Dropbox location to your local device and unzip, cd into the folder.
https://www.dropbox.com/s/32d7qyngv57zly9/prediction.zip?dl=0

2. Since we named our redis container "redis1", we will need to export this variable name to let the container use it. `export REDISURL='redis1'`

3. Build the docker container `docker build -t proj_final -f Dockerfile.final .`

4. Bring up the container
`docker run -it --rm --name=proj --network hw03 -e REDISURL=$REDISURL --hostname="proj" --link redis1:redis -h redis -p 6379 --runtime nvidia  -p 5000:5000 -v /tmp/.X11-unix/:/tmp/.X11-unix  -v $PWD/models:/work   proj_final:latest`

5. Once you are in the container, run `python3 main.py`. This will load the model and weights and open the endpoint up at port 5000 for listening.

![predictionendpt](https://github.com/shwethacn/W251-Final-Project/blob/master/imgs/edge_predictionendpt.jpg)

## PART V : Alexa + Ngrok + OpenCV

In the third terminal window ...

1. Copy the folder "imagecapture.zip" from the Dropbox location to your local device and unzip, cd into the folder.
https://www.dropbox.com/s/eiiigwr0hybwjew/imagecapture.zip?dl=0

2. Configure the export parameters:
```
export DISPLAY=:0
export RTSP='rtsp://192.168.1.9:5540/ch0'
export REDISURL='redis1'
export PREDICTURL = 'http://192.168.1.18:5000/predict'
xhost + 
```

3. Build the docker container `sudo docker build -t flaskit -f Dockerfile .`

4. Bring up the container
`sudo docker run --privileged -it --rm --name=flaskapp  --network hw03 -e RTSP=$RTSP -e DISPLAY=$DISPLAY -e REDISURL=$REDISURL --link redis1:redis -h redis -p 6379 -v /tmp/.X11-unix/:/tmp/.X11-unix --volume $PWD:/home/work flaskit:latest /bin/bash`

5. Once you are in the container, run `./run.sh`. This will bring up the ngrok window. Note the following:
- Session Status = online
- Forwarding HTTPS address (i.e something like `https://1316874182e7.ngrok.io `)
You will enter this ngrok url in the Endpoint box in the Alexa developer skills portal.

![ngrok](https://github.com/shwethacn/W251-Final-Project/blob/master/imgs/edge_ngrok.jpg)