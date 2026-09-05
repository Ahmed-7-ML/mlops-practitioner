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