# SQRS API Server

Flask REST API and WebSocket server for SQRS system.

## Setup

1. Install dependencies (already in main requirements.txt):
```bash
pip install flask flask-cors flask-socketio
```

2. Run the API server:
```bash
python api/app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/config` - Get configuration
- `GET /api/metrics/current` - Current KPIs
- `GET /api/metrics/historical` - Historical metrics
- `GET /api/agents` - Get all agents
- `GET /api/agents/<id>` - Get specific agent
- `GET /api/assignments/active` - Active assignments
- `POST /api/routing/matrix` - Compute routing matrix
- `GET /api/constraints/dual` - Get dual variables
- `GET /api/policies/compare` - Policy comparison
- `POST /api/simulation/start` - Start simulation
- `POST /api/simulation/stop` - Stop simulation
- `GET /api/simulation/status` - Simulation status

## WebSocket Events

- `connect` - Client connects
- `metrics_update` - Real-time metrics updates
- `simulation_complete` - Simulation finished

## Integration

This API wraps the existing SQRS core logic without modifying it. All imports use relative paths to access:
- `config.py`
- `data/`
- `models/`
- `routing/`
- `simulation/`
- `evaluation/`

