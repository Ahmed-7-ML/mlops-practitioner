# mlops-practitioner

## Architecture:
```
mlops-practitioner/
├── README.md               # # the 3-command quickstart — updated every module
├── pyproject.toml
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
│
├── src/prodml/                  ← الباكدج الأساسي (M1)
│   ├── config.py
│   ├── logging_conf.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── export.py
│   └── api/
│       ├── main.py
│       └── schemas.py
│
├── tests/                       ← بيزيد مع كل موديول
├── docker/                      ← M1 (Dockerfile + docker-compose)
├── pipelines/                   ← M2 (DVC) + M3 (Airflow)
├── infra/                       ← M2 (Terraform)
├── .github/workflows/           ← M2 (CI + Continuous Training)
├── serving/                     ← M3 (BentoML, Triton, vLLM, nginx)
├── loadtest/                    ← M3 (Locust)
├── optimization/                ← M4 (benchmarks)
├── monitoring/                  ← M5 (Evidently, Prometheus, Grafana...)
├── reports/                     ← التقارير بتاعتك (بتتقيّم)
└── docs/
    ├── module-1.md
    ├── module-2.md
    ...
    └── module-5.md
```

## Development Workflow

```bash
# 1. Install (editable + dev dependencies)
uv pip install -e ".[dev]"

# 2. Lint
ruff check src tests && black --check src tests

# 3. Test
pytest -v --cov=src/prodml --cov-report=term-missing

# 4. Train
python -m prodml.train

# 5. Serve (later in Module 3)
uvicorn prodml.api.main:app --reload --port 8000
```

## Test FastAPI Endpoints
```
# Health
curl -s http://localhost:8000/health | jq

# Metadata
curl -s http://localhost:8000/metadata | jq

# Single predict
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":"66","DOLocationID":"262","trip_distance":8.07,"passenger_count":1.0}' | jq

# Batch predict
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"PULocationID":"66","DOLocationID":"262","trip_distance":8.07,"passenger_count":1.0},{"PULocationID":"198","DOLocationID":"56","trip_distance":4.08,"passenger_count":1.0}]}' | jq

# Validation error (should return 422)
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":"66","DOLocationID":"262","trip_distance":-5}' | jq
```
