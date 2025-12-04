uvicorn main:app --reload
source .venv/bin/activate

docker build -t test-booted .
  docker run -d -p 8080:8080 --env-file .env --name edge-detector-test edge-detector:latest
  docker logs -f <container id>

curl -X POST http://localhost:8080/detect -F "file=@test_image__temp__.jpg"

docker build -t test-booted .
docker run -p 8080:8080 test-booted

docker stop <container id>
docker rm -f <container id>

# Remove and recreate venv
rm -rf .venv
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
