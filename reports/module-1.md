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
| Mean Latency      | 53.54 ms   | 4.59 ms   | **ONNX** (~11.7× faster)|
| Median Latency    | 51.38 ms   | 4.21 ms   | **ONNX**                |
| p95 Latency       | 70.74 ms   | 5.81 ms   | **ONNX** (~12.2× faster)|
| p99 Latency       | 103.52 ms  | 6.13 ms   | **ONNX**                |
| Max Diff          | —          | 9e-06     | Parity ✅               |
| Parity Test       | —          | PASSED    | ✅                      |

---

## Serialization Format Comparison

| Format     | Human-readable | Cross-language | Schema-enforced | Safe to load from untrusted source      |
|------------|----------------|----------------|------------------|-----------------------------------------|
| JSON       | ✅ Yes         | ✅ Yes         | ❌ No            | ✅ Yes                                  |
| Protobuf   | ❌ No          | ✅ Yes         | ✅ Yes           | ✅ Yes                                  |
| Pickle     | ❌ No          | ❌ No          | ❌ No            | ❌ **No** (executes arbitrary code)     |
| ONNX       | ❌ No          | ✅ Yes         | ✅ Yes           | ✅ Yes                                  |

> **Decision:** Our service will serve predictions using **ONNX** because it is cross-language, schema-enforced, safe to load from untrusted sources, and significantly faster than Pickle (≈12× lower p95 latency), while completely avoiding Pickle’s arbitrary code execution risk.
