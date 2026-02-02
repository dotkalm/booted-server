# Booted v1 (backend)

Companion to <a href="https://github.com/dotkalm/BootedWebAppFrontend/tree/main"> Booted Frontend </a>

<table>
<tr>
<td width="30%">
  <img src="https://github.com/user-attachments/assets/34473723-04a6-4008-98ec-0b4933c179a7" width="100%" alt="booted demo"/>
</td>
<td>
  <h3>At a glance</h3>
  
  Image recognition API for detecting coordinates of car wheels and gathering geometry heuristics so that the <a href="https://github.com/dotkalm/BootedWebAppFrontend/tree/main"> front end</a> can correctly overlay a 3d object onto an image. 
</td>
</tr>
</table>


## How its trained? 

<img src="https://github.com/user-attachments/assets/16e793d2-7bd4-4ddf-9b0d-f4bbb6b2d182" width="50%" />

Created a dataset using Roboflow

Used Ultralytics's Yolo library to train my model on the dataset I annotated in roboflow.

./tests contains my eval suite to validate successful training.

Yolo comes pre-loaded with car detection, so we run our car wheel detection model after cars are detected. 


## In Progress 
We want to use openCV's homography methods to determine the camera's perspective so that the front end can properly callibrate threeJS's wordview to the oringal photograph.

### local dev
<table>
<tr>
<td>
run in uv 
</td>
<td>
  
```uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload```
</td>
</tr>

<tr>
  <td>
    docker set up
  </td>
<td>
  
  ```docker build -t test-booted . ``` <br/>
  ``` docker run -d -p 8080:8080 --env-file .env --name edge-detector-test edge-detector:latest```<br/>
  ```  docker logs -f <container id> ```
</td>
</tr>
<tr>
  <td>
    local testing
  </td>
  <td>
    
     curl -X POST http://localhost:8080/detect -F "file=@test_image__temp__.jpg" 
  </td>
</tr>
</table>

## Run the docker container locally

  ```
  docker build -t edge-detector-api .

  docker images | grep edge-detector-api

  docker run -d -p 8080:8080 --name edge-detector edge-detector-api

  docker ps

  docker logs -f edge-detector
  ```
