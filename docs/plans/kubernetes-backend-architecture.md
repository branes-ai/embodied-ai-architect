# Kubernetes Backend Architecture

**Date**: 2025-11-02
**Status**: Design Proposal

## Overview

The Kubernetes backend enables cloud-native benchmarking with:
- **Horizontal scaling**: Run multiple benchmarks in parallel
- **Resource management**: GPU/CPU allocation per job
- **Isolation**: Each benchmark runs in its own container
- **Elasticity**: Scale up/down based on workload
- **Cost optimization**: Only pay for compute time used

## Use Cases

1. **Parallel Benchmarking**: Test model on 10 different hardware configs simultaneously
2. **Hyperparameter Sweeps**: Run 100 variants with different quantization settings
3. **Large-scale Testing**: Benchmark fleet of models across cluster
4. **GPU Allocation**: Request specific GPU types (V100, A100, T4)
5. **Multi-region**: Deploy to clusters in different regions for latency testing

## Architecture

### Job-Based Execution Pattern

```
BenchmarkAgent
    ↓
KubernetesBackend
    ↓
    ├─→ Create ConfigMap (model data)
    ├─→ Create Job (benchmark container)
    ├─→ Monitor Job status
    ├─→ Retrieve results (from pod logs)
    └─→ Cleanup (delete job, configmap)
```

**Why Jobs, not Deployments?**
- Jobs are ephemeral (cleanup automatic)
- Natural fit for batch workloads
- Built-in completion tracking
- No need for service exposure

### Horizontal Scaling Pattern

```
KubernetesBackend.execute_parallel([
    (model_1, config_1),
    (model_2, config_2),
    (model_3, config_3),
    ...
])
    ↓
Create N Jobs in parallel
    ↓
Monitor all jobs
    ↓
Collect all results
    ↓
Return aggregated results
```

## Kubernetes Resources

### 1. Job Specification

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: benchmark-{workflow_id}-{job_id}
  namespace: embodied-ai
  labels:
    app: embodied-ai-benchmark
    workflow: {workflow_id}
    backend: kubernetes
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 3600  # Auto-cleanup after 1 hour
  template:
    metadata:
      labels:
        app: embodied-ai-benchmark
        job: {job_id}
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: embodied-ai-benchmark:latest
        command: ["python", "/app/benchmark_runner.py"]
        env:
        - name: WORKFLOW_ID
          value: "{workflow_id}"
        - name: MODEL_PATH
          value: "/models/model.pt"
        volumeMounts:
        - name: model-data
          mountPath: /models
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"  # Request GPU
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
      volumes:
      - name: model-data
        configMap:
          name: model-{workflow_id}
      nodeSelector:
        gpu-type: nvidia-a100  # Target specific GPU
```

### 2. ConfigMap for Model Data

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-{workflow_id}
  namespace: embodied-ai
data:
  model.pt: |
    {base64_encoded_model}
  config.json: |
    {
      "input_shape": [1, 3, 224, 224],
      "iterations": 100,
      "warmup_iterations": 10
    }
```

### 3. Service Account (Security)

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: embodied-ai-benchmark
  namespace: embodied-ai
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: embodied-ai-benchmark
  namespace: embodied-ai
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "list", "delete"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["create", "get", "delete"]
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: embodied-ai-benchmark
  namespace: embodied-ai
subjects:
- kind: ServiceAccount
  name: embodied-ai-benchmark
roleRef:
  kind: Role
  name: embodied-ai-benchmark
  apiGroup: rbac.authorization.k8s.io
```

## Implementation Design

### KubernetesBackend Class

```python
class KubernetesBackend(BenchmarkBackend):
    """Execute benchmarks as Kubernetes Jobs.

    Features:
    - Parallel execution (N jobs at once)
    - GPU allocation
    - Resource limits
    - Automatic cleanup
    - Result retrieval from logs
    """

    def __init__(
        self,
        namespace: str = "embodied-ai",
        image: str = "embodied-ai-benchmark:latest",
        secrets_manager: SecretsManager = None,
        kubeconfig_secret: str = "k8s_kubeconfig"
    ):
        """Initialize Kubernetes backend."""
        self.namespace = namespace
        self.image = image
        self.secrets_manager = secrets_manager

        # Load kubeconfig from secrets
        kubeconfig_path = secrets_manager.get_secret(kubeconfig_secret)
        config.load_kube_config(config_file=kubeconfig_path)

        self.batch_api = client.BatchV1Api()
        self.core_api = client.CoreV1Api()

    def execute_benchmark(
        self,
        model: nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup_iterations: int = 10,
        config: Dict[str, Any] | None = None
    ) -> BenchmarkResult:
        """Execute single benchmark as K8s Job."""

        job_id = str(uuid.uuid4())[:8]

        # 1. Create ConfigMap with model
        configmap = self._create_model_configmap(model, job_id)

        # 2. Create Job
        job = self._create_benchmark_job(job_id, input_shape, iterations)

        # 3. Wait for completion
        self._wait_for_job_completion(job_id, timeout=600)

        # 4. Get results from logs
        result = self._get_job_results(job_id)

        # 5. Cleanup
        self._cleanup_job(job_id)

        return result

    def execute_parallel(
        self,
        benchmarks: List[Tuple[nn.Module, Dict]]
    ) -> List[BenchmarkResult]:
        """Execute multiple benchmarks in parallel.

        This is where horizontal scaling shines!
        """
        job_ids = []

        # Create all jobs
        for model, config in benchmarks:
            job_id = str(uuid.uuid4())[:8]
            self._create_model_configmap(model, job_id)
            self._create_benchmark_job(job_id, config)
            job_ids.append(job_id)

        # Wait for all to complete (in parallel)
        results = []
        for job_id in job_ids:
            self._wait_for_job_completion(job_id)
            result = self._get_job_results(job_id)
            results.append(result)
            self._cleanup_job(job_id)

        return results
```

## Resource Management

### GPU Types and Node Selectors

```python
# Target specific GPU types
gpu_configs = {
    "nvidia-v100": {
        "nodeSelector": {"gpu-type": "nvidia-v100"},
        "resources": {"nvidia.com/gpu": "1"}
    },
    "nvidia-a100": {
        "nodeSelector": {"gpu-type": "nvidia-a100"},
        "resources": {"nvidia.com/gpu": "1"}
    },
    "nvidia-t4": {
        "nodeSelector": {"gpu-type": "nvidia-t4"},
        "resources": {"nvidia.com/gpu": "1"}
    }
}

# Use in backend
backend = KubernetesBackend(
    gpu_type="nvidia-a100",
    gpu_count=2  # Multi-GPU
)
```

### Resource Requests and Limits

```python
# Conservative (cost-effective)
resources_small = {
    "requests": {"memory": "2Gi", "cpu": "1"},
    "limits": {"memory": "4Gi", "cpu": "2"}
}

# Aggressive (performance)
resources_large = {
    "requests": {"memory": "16Gi", "cpu": "8", "nvidia.com/gpu": "4"},
    "limits": {"memory": "32Gi", "cpu": "16", "nvidia.com/gpu": "4"}
}
```

## Scaling Patterns

### 1. Fixed Parallelism

```python
# Run 10 benchmarks in parallel
backend = KubernetesBackend(max_parallel_jobs=10)

results = backend.execute_parallel([
    (model1, config1),
    (model2, config2),
    # ... 10 total
])
```

### 2. Dynamic Scaling

```python
# Scale based on cluster capacity
backend = KubernetesBackend(auto_scale=True)

# Will automatically batch based on available nodes
results = backend.execute_parallel(
    [(model, config) for model, config in model_configs]
)
```

### 3. Priority-Based

```python
# High-priority jobs get scheduled first
job_specs = [
    {"model": model1, "config": config1, "priority": "high"},
    {"model": model2, "config": config2, "priority": "low"},
]

backend.execute_with_priorities(job_specs)
```

## Container Image

### Dockerfile for Benchmark Runner

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy benchmark runner
COPY benchmark_runner.py .
COPY utils/ ./utils/

# User for security
RUN useradd -m -u 1000 benchmark
USER benchmark

ENTRYPOINT ["python", "benchmark_runner.py"]
```

### Benchmark Runner Script

```python
# benchmark_runner.py
import os
import json
import torch
import time
from pathlib import Path

def main():
    # Load model from configmap
    model_path = Path(os.getenv("MODEL_PATH"))
    model = torch.load(model_path)

    # Load config
    config_path = Path("/models/config.json")
    config = json.loads(config_path.read_text())

    # Run benchmark
    results = benchmark(model, config)

    # Output results (will be captured from logs)
    print(json.dumps(results))

if __name__ == "__main__":
    main()
```

## Security Considerations

### 1. Kubeconfig Handling

```python
# Use SecretsManager for kubeconfig
secrets = SecretsManager([
    EnvironmentSecretsProvider(),
    FileSecretsProvider("config/credentials")
])

# Kubeconfig stored securely
backend = KubernetesBackend(
    secrets_manager=secrets,
    kubeconfig_secret="k8s_kubeconfig"
)
```

### 2. Service Account Permissions

- **Least Privilege**: Only create/delete jobs in specific namespace
- **No cluster-wide access**: Scoped to namespace
- **RBAC**: Explicit permissions via Role/RoleBinding

### 3. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: embodied-ai-benchmark
  namespace: embodied-ai
spec:
  podSelector:
    matchLabels:
      app: embodied-ai-benchmark
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 443  # API server only
```

## Cost Optimization

### 1. Spot Instances

```yaml
spec:
  template:
    spec:
      nodeSelector:
        node-type: spot  # Use spot instances (cheaper)
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### 2. Auto-scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embodied-ai-benchmark
spec:
  scaleTargetRef:
    apiVersion: batch/v1
    kind: Job
    name: embodied-ai-benchmark
  minReplicas: 1
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. TTL for Auto-cleanup

```yaml
spec:
  ttlSecondsAfterFinished: 3600  # Delete after 1 hour
  backoffLimit: 2  # Max retries
```

## Monitoring and Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

jobs_created = Counter('benchmark_jobs_created', 'Jobs created')
job_duration = Histogram('benchmark_job_duration', 'Job duration')
job_failures = Counter('benchmark_job_failures', 'Job failures')

@job_duration.time()
def run_benchmark():
    jobs_created.inc()
    try:
        result = backend.execute_benchmark(...)
        return result
    except Exception:
        job_failures.inc()
        raise
```

### Logging

```python
# Structured logging
logger.info("job.created", extra={
    "job_id": job_id,
    "namespace": namespace,
    "gpu_type": gpu_type,
    "workflow_id": workflow_id
})
```

## Example Workflows

### 1. GPU Comparison

```python
# Compare same model across different GPUs
backend = KubernetesBackend(namespace="embodied-ai")

gpu_types = ["nvidia-v100", "nvidia-a100", "nvidia-t4"]
results = {}

for gpu_type in gpu_types:
    backend.set_gpu_type(gpu_type)
    result = backend.execute_benchmark(model, input_shape)
    results[gpu_type] = result

# Generate comparison report
print(f"V100: {results['nvidia-v100'].mean_latency_ms}ms")
print(f"A100: {results['nvidia-a100'].mean_latency_ms}ms")
print(f"T4: {results['nvidia-t4'].mean_latency_ms}ms")
```

### 2. Hyperparameter Sweep

```python
# Test 100 different quantization configs
configs = [
    {"quantization": "int8", "precision": "fp16"},
    {"quantization": "int4", "precision": "fp16"},
    # ... 100 total
]

# Run all in parallel on K8s
results = backend.execute_parallel([
    (model, config) for config in configs
])

# Find best config
best = min(results, key=lambda r: r.mean_latency_ms)
print(f"Best config: {best.metadata['config']}")
```

### 3. Multi-Model Fleet

```python
# Benchmark entire model zoo
models = load_model_zoo()  # 50 models

# Batch into groups of 10 (cluster capacity)
for batch in chunk(models, 10):
    results = backend.execute_parallel([
        (model, default_config) for model in batch
    ])

    # Store results
    for result in results:
        db.store_result(result)
```

## Advantages over SSH Backend

| Feature | SSH Backend | Kubernetes Backend |
|---------|-------------|-------------------|
| Scalability | Limited by single machine | Horizontal (100s of nodes) |
| Isolation | None | Container-level |
| GPU Management | Manual | Automatic allocation |
| Cleanup | Manual | Automatic (TTL) |
| Cost | Fixed | Pay-per-use |
| Reliability | Single point of failure | Job retries, HA |
| Multi-tenancy | Difficult | Built-in namespaces |

## Best Practices

1. **Use Jobs, not Pods**: Jobs have built-in completion tracking
2. **Set TTL**: Automatic cleanup after completion
3. **Resource Limits**: Prevent runaway jobs
4. **Namespace Isolation**: Separate dev/staging/prod
5. **Node Selectors**: Target specific hardware
6. **ConfigMaps**: Don't bake data into images
7. **Monitoring**: Prometheus + Grafana
8. **Logging**: Structured logs to ELK/Loki

## Future Enhancements

1. **Spot Instance Support**: Use preemptible nodes for cost savings
2. **Job Prioritization**: High-priority jobs first
3. **Result Storage**: Push to S3/GCS instead of logs
4. **Distributed Training**: Multi-node PyTorch DDP
5. **Auto-scaling**: HPA based on queue depth
6. **Regional Deployment**: Run in multiple regions
