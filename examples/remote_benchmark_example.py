"""Example demonstrating secure remote benchmarking with SSH backend.

This example shows:
- How to use SecretsManager for secure credential handling
- How to configure RemoteSSHBackend with secrets
- How secrets are never hardcoded or logged
- Proper error handling and cleanup

Setup Instructions:
1. Copy .env.example to .env
2. Fill in your SSH credentials:
   EMBODIED_AI_SSH_HOST=your-server.example.com
   EMBODIED_AI_SSH_USER=your-username
   EMBODIED_AI_SSH_KEY=/path/to/your/private_key

3. Ensure your SSH key has correct permissions:
   chmod 600 /path/to/your/private_key

4. Install remote dependencies:
   pip install 'embodied-ai-architect[remote]'

5. Run the example:
   python examples/remote_benchmark_example.py
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

# Import remote backend (gracefully handle if not installed)
try:
    from embodied_ai_architect.agents.benchmark.backends import RemoteSSHBackend
    REMOTE_AVAILABLE = True
except ImportError:
    REMOTE_AVAILABLE = False
    print("⚠️  Remote benchmarking not available.")
    print("   Install with: pip install 'embodied-ai-architect[remote]'")


# Simple model for testing
class TinyModel(nn.Module):
    """Tiny model for testing remote benchmarking."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        return self.fc(x)


def main():
    """Demonstrate secure remote benchmarking."""

    print("=" * 70)
    print("Embodied AI Architect - Secure Remote Benchmarking Example")
    print("=" * 70)

    if not REMOTE_AVAILABLE:
        print("\n❌ This example requires remote dependencies.")
        print("   Install with: pip install 'embodied-ai-architect[remote]'")
        return

    # 1. Initialize Secrets Manager
    print("\n🔒 Initializing Secrets Manager...")

    # Try multiple providers in order of precedence:
    # 1. Environment variables (development)
    # 2. Files in config/credentials (production)
    secrets_manager = SecretsManager([
        EnvironmentSecretsProvider(prefix="EMBODIED_AI_"),
        FileSecretsProvider("config/credentials")
    ])

    # List available secrets (keys only, not values!)
    print(f"   Available secret providers: {len(secrets_manager.providers)}")
    all_keys = secrets_manager.list_all_keys()
    for provider, keys in all_keys.items():
        if keys:
            print(f"   {provider}: {len(keys)} secrets available")

    # 2. Check if SSH credentials are configured
    print("\n🔑 Checking SSH credentials...")

    try:
        # Try to get SSH configuration (will raise if not found)
        ssh_host = secrets_manager.get_secret("ssh_host", required=False)
        ssh_user = secrets_manager.get_secret("ssh_user", required=False)
        ssh_key = secrets_manager.get_secret("ssh_key", required=False)

        if not all([ssh_host, ssh_user, ssh_key]):
            print("\n⚠️  SSH credentials not configured.")
            print("\n   To use remote benchmarking:")
            print("   1. Copy .env.example to .env")
            print("   2. Set these variables:")
            print("      EMBODIED_AI_SSH_HOST=your-server.example.com")
            print("      EMBODIED_AI_SSH_USER=your-username")
            print("      EMBODIED_AI_SSH_KEY=/path/to/your/private_key")
            print("   3. Ensure key has correct permissions: chmod 600 <key_file>")
            print("\n   Running in demo mode (simulation)...")

            # Demo mode - show what would happen
            demonstrate_security_features(secrets_manager)
            return

        print(f"   ✓ SSH Host: {ssh_host}")
        print(f"   ✓ SSH User: {ssh_user}")
        print(f"   ✓ SSH Key: [REDACTED]")  # Never print the actual key!

        # 3. Create model
        print("\n📦 Creating test model...")
        model = TinyModel()
        print(f"   Model: {type(model).__name__}")
        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params:,}")

        # 4. Create Remote SSH Backend
        print(f"\n🌐 Connecting to remote host: {ssh_host}...")

        backend = RemoteSSHBackend(
            host=ssh_host,
            user=ssh_user,
            secrets_manager=secrets_manager,
            ssh_key_secret="ssh_key",
            port=22
        )

        # Check if remote is available
        if not backend.is_available():
            print(f"\n❌ Cannot connect to {ssh_host}")
            print("   Check that:")
            print("   - Host is reachable")
            print("   - SSH key is correct")
            print("   - User has access")
            return

        print(f"   ✓ Connection successful!")

        # 5. Run remote benchmark
        print(f"\n⚡ Running benchmark on {ssh_host}...")
        print("   (This may take a minute...)")

        result = backend.execute_benchmark(
            model=model,
            input_shape=(1, 10),
            iterations=50,
            warmup_iterations=5
        )

        # 6. Display results
        print("\n📊 Benchmark Results:")
        print(f"   Backend: {result.backend_name}")
        print(f"   Device: {result.device}")
        print(f"   Mean Latency: {result.mean_latency_ms:.3f} ms")
        print(f"   Std Dev: {result.std_latency_ms:.3f} ms")
        print(f"   Min/Max: {result.min_latency_ms:.3f} / {result.max_latency_ms:.3f} ms")
        print(f"   Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")

        # 7. Show audit log (demonstrates security tracking)
        print("\n📝 Security Audit Log:")
        audit = secrets_manager.get_audit_log()
        for entry in audit:
            # Mask sensitive data in audit log
            print(f"   [{entry['timestamp']}] {entry['action']}: {entry['key']}")

        print("\n✅ Remote benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Note: Exception messages are automatically masked by secrets manager
        print("\n   Make sure your .env file is configured correctly.")


def demonstrate_security_features(secrets_manager):
    """Demonstrate security features without actual remote connection.

    Args:
        secrets_manager: SecretsManager instance
    """
    print("\n" + "=" * 70)
    print("Security Features Demonstration")
    print("=" * 70)

    # 1. Secret masking
    print("\n1️⃣  Secret Masking:")
    fake_key = "/home/user/.ssh/id_rsa"
    message = f"Using SSH key: {fake_key}"
    masked = secrets_manager.mask_secret(message, fake_key)
    print(f"   Original: {message}")
    print(f"   Masked:   {masked}")

    # 2. Configuration with references
    print("\n2️⃣  Configuration References:")
    config = {
        "host": "${env:EMBODIED_AI_SSH_HOST}",
        "user": "${env:EMBODIED_AI_SSH_USER}",
        "key": "${secret:ssh_key}"
    }
    print(f"   Config with references: {config}")
    print("   (References would be resolved at runtime)")

    # 3. Audit logging
    print("\n3️⃣  Audit Logging:")
    print("   Every secret access is logged:")
    print("   - Who accessed it")
    print("   - When it was accessed")
    print("   - Which provider provided it")
    print("   - Whether it succeeded")

    # 4. Multi-provider hierarchy
    print("\n4️⃣  Multi-Provider Hierarchy:")
    print("   Secrets are tried in order:")
    print("   1. Environment variables (EMBODIED_AI_*)")
    print("   2. Files in config/credentials/")
    print("   3. Vault services (production)")
    print("   First match wins!")

    # 5. Secure defaults
    print("\n5️⃣  Secure Defaults:")
    print("   ✓ File permissions checked (600 or 400)")
    print("   ✓ Directory ownership validated")
    print("   ✓ Secrets never in logs or reports")
    print("   ✓ Error messages mask secrets")
    print("   ✓ No hardcoded credentials")

    print("\n" + "=" * 70)
    print("\n💡 To run actual remote benchmarks:")
    print("   1. Configure .env with your SSH credentials")
    print("   2. Run this script again")
    print("=" * 70)


if __name__ == "__main__":
    main()
