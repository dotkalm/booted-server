uvicorn main:app --reload
source .venv/bin/activate

  docker run -d -p 8080:8080 --env-file .env --name edge-detector-test edge-detector:latest
  docker logs -f <container id>

curl -X POST http://localhost:8080/detect -F "file=@test_image__temp__.jpg"

docker build -t test-booted .
docker run -p 8080:8080 test-booted
