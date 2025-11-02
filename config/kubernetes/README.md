# Kubernetes Configuration

This directory contains Kubernetes manifests for deploying the Embodied AI Architect benchmarking system.

## Quick Start

### 1. Setup Cluster

```bash
# Apply namespace
kubectl apply -f namespace.yaml

# Apply RBAC (service account, role, rolebinding)
kubectl apply -f rbac.yaml

# Verify
kubectl get sa -n embodied-ai
kubectl get role -n embodied-ai
```

### 2. Configure Access

```bash
# Get kubeconfig (save securely!)
kubectl config view --raw > ~/.kube/embodied-ai-config

# Set permissions
chmod 600 ~/.kube/embodied-ai-config

# Add to .env
echo "EMBODIED_AI_K8S_KUBECONFIG=$HOME/.kube/embodied-ai-config" >> .env
```

### 3. Test Connection

```python
from embodied_ai_architect.agents.benchmark.backends import KubernetesBackend
from embodied_ai_architect.security import SecretsManager, EnvironmentSecretsProvider

secrets = SecretsManager([EnvironmentSecretsProvider()])
backend = KubernetesBackend(
    namespace="embodied-ai",
    secrets_manager=secrets,
    kubeconfig_secret="k8s_kubeconfig"
)

print(f"Cluster accessible: {backend.is_available()}")
```

## Files

- `namespace.yaml`: Creates `embodied-ai` namespace
- `rbac.yaml`: Service account with minimal permissions
- `README.md`: This file

## Resource Requests

Default resource requests per job:
- CPU: 1 core (request), 2 cores (limit)
- Memory: 2Gi (request/limit)
- GPU: Optional (configure via `gpu_type` parameter)

## Node Selectors

Target specific GPU types:

```python
backend = KubernetesBackend(
    gpu_type="nvidia-a100",  # Node must have label: gpu-type=nvidia-a100
    cpu_request="4",
    memory_request="16Gi"
)
```

Ensure nodes are labeled:
```bash
kubectl label nodes gpu-node-1 gpu-type=nvidia-a100
kubectl label nodes gpu-node-2 gpu-type=nvidia-v100
```

## Security

### Least Privilege
The service account only has permissions to:
- Create/delete jobs and configmaps in `embodied-ai` namespace
- Read pod logs (for result retrieval)
- No cluster-admin or cross-namespace access

### Network Policies

For additional security, add network policy:

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
      port: 443
```

## Cost Optimization

### Use Spot Instances

```python
backend = KubernetesBackend(
    namespace="embodied-ai",
    node_selector={"node-type": "spot"}
)
```

Ensure spot nodes are labeled:
```bash
kubectl label nodes spot-node-1 node-type=spot
```

### TTL for Auto-cleanup

Jobs automatically cleanup after completion (default: 1 hour):

```python
backend = KubernetesBackend(
    ttl_seconds=3600  # Delete job 1 hour after completion
)
```

## Monitoring

### View Running Jobs

```bash
kubectl get jobs -n embodied-ai
kubectl get pods -n embodied-ai
```

### View Job Logs

```bash
# Get pod name
POD=$(kubectl get pods -n embodied-ai -l job=<job-id> -o name | head -1)

# View logs
kubectl logs -n embodied-ai $POD
```

### Cleanup All Jobs

```bash
# Delete all completed jobs
kubectl delete jobs -n embodied-ai --field-selector status.successful=1

# Delete all jobs
kubectl delete jobs -n embodied-ai --all
```

## Troubleshooting

### Job Stuck in Pending

```bash
# Check events
kubectl describe job benchmark-<job-id> -n embodied-ai

# Common issues:
# - Insufficient resources (CPU/memory/GPU)
# - Node selector doesn't match any nodes
# - Image pull errors
```

### Out of Memory

Increase memory request:
```python
backend = KubernetesBackend(memory_request="8Gi")
```

### GPU Not Allocated

Check:
1. Node has GPU: `kubectl describe node <node>`
2. GPU device plugin installed
3. Node has correct label: `gpu-type=nvidia-a100`

## Advanced Usage

### Multiple Namespaces

Use different namespaces for dev/staging/prod:

```python
dev_backend = KubernetesBackend(namespace="embodied-ai-dev")
prod_backend = KubernetesBackend(namespace="embodied-ai-prod")
```

### Custom Images

Build and push your own benchmark image:

```bash
docker build -t my-registry/embodied-ai-benchmark:v1 .
docker push my-registry/embodied-ai-benchmark:v1
```

Use custom image:
```python
backend = KubernetesBackend(
    image="my-registry/embodied-ai-benchmark:v1"
)
```

### Resource Quotas

Limit total resources in namespace:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: embodied-ai-quota
  namespace: embodied-ai
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    requests.nvidia.com/gpu: "10"
```
