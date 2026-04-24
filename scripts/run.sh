#!/bin/bash
echo "Starting Database & Cache..."
docker-compose up -d

echo "Starting FastAPI Server..."
uvicorn app.main:app --reload --port 8000
