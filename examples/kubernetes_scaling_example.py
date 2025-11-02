"""Example demonstrating Kubernetes horizontal scaling.

This example shows how to use the Kubernetes backend to:
1. Run benchmarks on cloud infrastructure
2. Scale horizontally (run many benchmarks in parallel)
3. Compare multiple hardware configurations simultaneously
4. Optimize for cost and performance

Setup Instructions:
1. Deploy Kubernetes resources:
   kubectl apply -f config/kubernetes/namespace.yaml
   kubectl apply -f config/kubernetes/rbac.yaml

2. Configure kubeconfig:
   kubectl config view --raw > ~/.kube/embodied-ai-config
   chmod 600 ~/.kube/embodied-ai-config

3. Add to .env:
   EMBODIED_AI_K8S_KUBECONFIG=/path/to/embodied-ai-config

4. Install kubernetes dependencies:
   pip install 'embodied-ai-architect[kubernetes]'

5. Run the example:
   python examples/kubernetes_scaling_example.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from embodied_ai_architect.security import (
    SecretsManager,
    EnvironmentSecretsProvider,
    FileSecretsProvider
)

# Import Kubernetes backend (gracefully handle if not installed)
try:
    from embodied_ai_architect.agents.benchmark.backends import KubernetesBackend
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    print("⚠️  Kubernetes backend not available.")
    print("   Install with: pip install 'embodied-ai-architect[kubernetes]'")


# Define test models
class TinyModel(nn.Module):
    """Tiny model for quick tests."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        return self.fc(x)


class SmallCNN(nn.Module):
    """Small CNN for image tasks."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def single_benchmark_example():
    """Example 1: Single benchmark on Kubernetes."""

    print("\n" + "=" * 70)
    print("Example 1: Single Benchmark on Kubernetes")
    print("=" * 70)

    secrets = SecretsManager([
        EnvironmentSecretsProvider(),
        FileSecretsProvider("config/credentials")
    ])

    backend = KubernetesBackend(
        namespace="embodied-ai",
        secrets_manager=secrets,
        kubeconfig_secret="k8s_kubeconfig",
        cpu_request="2",
        memory_request="4Gi"
    )

    if not backend.is_available():
        print("\n❌ Kubernetes cluster not accessible")
        print("   Check kubeconfig and cluster status")
        return

    print("✓ Kubernetes cluster accessible")

    model = TinyModel()
    print(f"\nBenchmarking {type(model).__name__}...")

    result = backend.execute_benchmark(
        model=model,
        input_shape=(1, 10),
        iterations=50,
        warmup_iterations=5
    )

    print(f"\n✓ Results:")
    print(f"   Mean Latency: {result.mean_latency_ms:.3f}ms")
    print(f"   Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")


def parallel_benchmark_example():
    """Example 2: Parallel benchmarks (horizontal scaling)."""

    print("\n" + "=" * 70)
    print("Example 2: Parallel Benchmarks (Horizontal Scaling)")
    print("=" * 70)

    secrets = SecretsManager([EnvironmentSecretsProvider()])

    backend = KubernetesBackend(
        namespace="embodied-ai",
        secrets_manager=secrets,
        kubeconfig_secret="k8s_kubeconfig"
    )

    if not backend.is_available():
        print("\n❌ Kubernetes cluster not accessible")
        return

    # Create multiple model variants
    models_and_configs = [
        (TinyModel(), {"input_shape": (1, 10), "iterations": 50}),
        (TinyModel(), {"input_shape": (1, 10), "iterations": 100}),
        (TinyModel(), {"input_shape": (1, 10), "iterations": 200}),
        (SmallCNN(), {"input_shape": (1, 3, 32, 32), "iterations": 50}),
        (SmallCNN(), {"input_shape": (1, 3, 64, 64), "iterations": 50}),
    ]

    print(f"\n🚀 Running {len(models_and_configs)} benchmarks in parallel...")
    print("   This demonstrates horizontal scaling on Kubernetes!")

    results = backend.execute_parallel(models_and_configs)

    print(f"\n✓ Completed {len(results)} benchmarks:")
    for i, result in enumerate(results):
        print(f"   {i+1}. {result.device}: {result.mean_latency_ms:.3f}ms")


def gpu_comparison_example():
    """Example 3: Compare different GPU types in parallel."""

    print("\n" + "=" * 70)
    print("Example 3: GPU Comparison (Parallel across GPU types)")
    print("=" * 70)

    secrets = SecretsManager([EnvironmentSecretsProvider()])

    gpu_types = ["nvidia-v100", "nvidia-a100", "nvidia-t4"]

    print(f"\nComparing model performance across {len(gpu_types)} GPU types...")
    print("All benchmarks run in parallel!")

    model = SmallCNN()
    results = {}

    for gpu_type in gpu_types:
        backend = KubernetesBackend(
            namespace="embodied-ai",
            secrets_manager=secrets,
            kubeconfig_secret="k8s_kubeconfig",
            gpu_type=gpu_type
        )

        if backend.is_available():
            result = backend.execute_benchmark(
                model=model,
                input_shape=(1, 3, 224, 224),
                iterations=100
            )
            results[gpu_type] = result

    if results:
        print(f"\n✓ GPU Comparison Results:")
        for gpu_type, result in sorted(results.items(), key=lambda x: x[1].mean_latency_ms):
            print(f"   {gpu_type}: {result.mean_latency_ms:.3f}ms")
            print(f"      Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
    else:
        print("\n⚠️  No results (GPUs may not be available in cluster)")


def hyperparameter_sweep_example():
    """Example 4: Hyperparameter sweep using Kubernetes scaling."""

    print("\n" + "=" * 70)
    print("Example 4: Hyperparameter Sweep (10 configurations)")
    print("=" * 70)

    secrets = SecretsManager([EnvironmentSecretsProvider()])

    backend = KubernetesBackend(
        namespace="embodied-ai",
        secrets_manager=secrets,
        kubeconfig_secret="k8s_kubeconfig",
        cpu_request="2",
        memory_request="4Gi"
    )

    if not backend.is_available():
        print("\n❌ Kubernetes cluster not accessible")
        return

    # Generate parameter sweep
    input_sizes = [16, 32, 64, 128, 224]
    batch_sizes = [1, 4]

    benchmarks = []
    for input_size in input_sizes:
        for batch_size in batch_sizes:
            model = SmallCNN()
            config = {
                "input_shape": (batch_size, 3, input_size, input_size),
                "iterations": 50,
                "warmup_iterations": 5
            }
            benchmarks.append((model, config))

    print(f"\n🔬 Running hyperparameter sweep with {len(benchmarks)} configurations...")
    print("   All running in parallel on Kubernetes!")

    results = backend.execute_parallel(benchmarks)

    print(f"\n✓ Sweep completed:")
    for i, ((model, config), result) in enumerate(zip(benchmarks, results)):
        input_shape = config["input_shape"]
        print(f"   {i+1}. Input {input_shape}: {result.mean_latency_ms:.3f}ms")


def demonstrate_benefits():
    """Demonstrate benefits of Kubernetes backend."""

    print("\n" + "=" * 70)
    print("Kubernetes Backend Benefits")
    print("=" * 70)

    benefits = {
        "Horizontal Scaling": [
            "Run 10, 100, or 1000 benchmarks in parallel",
            "Limited only by cluster size",
            "Automatically distributed across nodes"
        ],
        "Resource Management": [
            "Request specific GPUs (V100, A100, T4)",
            "Set CPU/memory limits per job",
            "Node selectors for hardware targeting"
        ],
        "Cost Optimization": [
            "Use spot/preemptible instances",
            "Pay only for compute time used",
            "Auto-cleanup with TTL"
        ],
        "Isolation": [
            "Each benchmark in own container",
            "No interference between jobs",
            "Clean environment per run"
        ],
        "Elasticity": [
            "Scale cluster up/down based on workload",
            "Auto-scaling with HPA",
            "Multi-region deployment"
        ]
    }

    for category, points in benefits.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  • {point}")

    print("\n" + "=" * 70)
    print("\n💡 Use Cases:")
    print("   • Model zoo benchmarking (100s of models)")
    print("   • Hyperparameter optimization sweeps")
    print("   • Multi-GPU type comparison")
    print("   • CI/CD performance testing at scale")
    print("   • A/B testing different model architectures")


def main():
    """Run all examples."""

    print("=" * 70)
    print("Embodied AI Architect - Kubernetes Scaling Examples")
    print("=" * 70)

    if not K8S_AVAILABLE:
        print("\n❌ Kubernetes backend not available")
        print("   Install with: pip install 'embodied-ai-architect[kubernetes]'")
        print("\n   Showing benefits demonstration instead:")
        demonstrate_benefits()
        return

    # Check if kubeconfig is configured
    secrets = SecretsManager([
        EnvironmentSecretsProvider(),
        FileSecretsProvider("config/credentials")
    ])

    kubeconfig = secrets.get_secret("k8s_kubeconfig", required=False)

    if not kubeconfig:
        print("\n⚠️  Kubeconfig not configured")
        print("\n   To use Kubernetes backend:")
        print("   1. Apply K8s manifests: kubectl apply -f config/kubernetes/")
        print("   2. Get kubeconfig: kubectl config view --raw > ~/.kube/embodied-ai-config")
        print("   3. Set permissions: chmod 600 ~/.kube/embodied-ai-config")
        print("   4. Add to .env: EMBODIED_AI_K8S_KUBECONFIG=/path/to/config")
        print("\n   Showing benefits demonstration instead:")
        demonstrate_benefits()
        return

    # Run examples
    try:
        single_benchmark_example()
        parallel_benchmark_example()
        gpu_comparison_example()
        hyperparameter_sweep_example()

        print("\n" + "=" * 70)
        print("✅ All examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n   Check:")
        print("   - Kubernetes cluster is accessible")
        print("   - Namespace 'embodied-ai' exists")
        print("   - RBAC is configured")
        print("   - kubeconfig has correct permissions")


if __name__ == "__main__":
    main()
