### Build and Run

oedisi build -m 

cd build/<UUID_FOLDER>

# Build Docker images
docker compose build

# Start containers
docker compose up -d

# Configure simulation (POST system.json to broker)
curl -X POST http://localhost:8766/configure -H "Content-Type: application/json" -d @../../system.json

# Run simulation
curl -X POST http://localhost:8766/run


# Download results via API
curl http://localhost:8766/results -o results.zip
```

