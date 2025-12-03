uvicorn main:app --reload
source .venv/bin/activate

  docker run -d -p 8080:8080 --env-file .env --name edge-detector-test edge-detector:latest
  docker logs -f <container id>
