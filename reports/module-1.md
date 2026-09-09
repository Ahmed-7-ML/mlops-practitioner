# Module 1 Report

## Baseline Results

**Chosen Model:** Linear Regression

| Metric | Value |
|--------|-------|
| RMSE   | 4.44  |
| MAE    | 2.98  |
| R²     | 0.76  |

---

## Serialization: Pickle vs ONNX

### Latency Benchmark (500 samples)

| Metric            | Pickle     | ONNX      | Winner                  |
|-------------------|------------|-----------|-------------------------|
| Mean Latency      | 5.36 ms   | 0.78 ms   | **ONNX** (~6.9× faster)|
| Median Latency    | 3.50 ms   | 0.33 ms   | **ONNX** (~10.6× faster)|
| p95 Latency       | 14.25 ms   | 2.74 ms   | **ONNX** (~5.2× faster)|
| p99 Latency       | 18.97 ms  | 7.13 ms   | **ONNX** (~2.7× faster)|
| Max Diff          | —          | 4e-06     | Parity ✅               |
| Parity Test       | —          | PASSED    | ✅                      |

---
## Serialization Format Comparison

| Format     | Human-readable | Cross-language | Schema-enforced | Safe to load from untrusted source      |
|------------|----------------|----------------|------------------|-----------------------------------------|
| JSON       | ✅ Yes         | ✅ Yes         | ❌ No            | ✅ Yes                                  |
| Protobuf   | ❌ No          | ✅ Yes         | ✅ Yes           | ✅ Yes                                  |
| Pickle     | ❌ No          | ❌ No          | ❌ No            | ❌ **No** (executes arbitrary code)     |
| ONNX       | ❌ No          | ✅ Yes         | ✅ Yes           | ✅ Yes                                  |

> **Decision:** Our service will serve predictions using **ONNX** because it is cross-language, schema-enforced, safe to load from untrusted sources, and significantly faster than Pickle (≈6.9× lower mean latency and ≈5.2× lower p95 latency), while completely avoiding Pickle’s arbitrary code execution risk.
