# Security Architecture - Secrets Management

**Date**: 2025-11-02
**Status**: Design Proposal

## Problem Statement

Remote benchmarking requires secure handling of:
- SSH private keys
- API tokens (AWS, GCP, Azure)
- Database credentials
- Service account keys
- Hardware vendor API keys
- Robot controller credentials

**Key Requirements**:
- Never commit secrets to git
- Support multiple secret sources (env vars, files, vault services)
- Principle of least privilege
- Audit logging
- Secure defaults
- Easy to use but hard to misuse

## Threat Model

### Threats We're Protecting Against

1. **Secret Leakage**
   - Accidental commit to version control
   - Logs containing secrets
   - Error messages exposing credentials
   - Report artifacts containing secrets

2. **Unauthorized Access**
   - Privilege escalation
   - Lateral movement (compromised backend shouldn't access all backends)
   - Replay attacks

3. **Insider Threats**
   - Minimize blast radius
   - Audit trail
   - Least privilege

### Threats Out of Scope (for now)

- Memory scraping attacks
- Side-channel attacks
- Physical access to machines
- Zero-day exploits

## Security Principles

1. **Defense in Depth**: Multiple layers of security
2. **Least Privilege**: Minimum permissions needed
3. **Fail Secure**: Errors should deny access, not grant it
4. **Secure by Default**: Safe configuration out of the box
5. **Separation of Concerns**: Secrets isolated from business logic
6. **Audit Everything**: Track who accessed what when

## Architecture

### 1. Secrets Provider Pattern

```
Backend needs credentials
    ↓
SecretsProvider Interface
    ├─→ EnvironmentSecretsProvider    (env vars)
    ├─→ FileSecretsProvider            (files with permissions)
    ├─→ VaultSecretsProvider           (HashiCorp Vault)
    ├─→ CloudSecretsProvider           (AWS Secrets Manager, GCP Secret Manager)
    └─→ KeyringSecretsProvider         (OS keychain)
```

**Interface**:
```python
class SecretsProvider(ABC):
    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        pass

    @abstractmethod
    def list_keys(self) -> List[str]:
        pass
```

### 2. Configuration Hierarchy

```
1. Environment Variables (highest priority)
   EMBODIED_AI_SSH_KEY=/path/to/key

2. Configuration File (.env, config.yaml)
   ssh:
     key_path: /path/to/key

3. System Keychain (OS credential storage)
   macOS: Keychain
   Linux: Secret Service API
   Windows: Credential Manager

4. Vault Service (production)
   HashiCorp Vault
   AWS Secrets Manager
   Azure Key Vault
```

### 3. Secret Types and Scoping

```python
class SecretScope(Enum):
    USER = "user"           # User-specific (development)
    BACKEND = "backend"     # Backend-specific (ssh_remote_1)
    GLOBAL = "global"       # System-wide (shared resources)
    EPHEMERAL = "ephemeral" # Temporary (session tokens)

class SecretType(Enum):
    SSH_KEY = "ssh_key"
    API_TOKEN = "api_token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
```

**Naming Convention**:
```
{scope}_{backend}_{type}_{identifier}

Examples:
- backend_ssh_remote_gpu_ssh_key_private
- global_aws_api_token
- user_dev_ssh_key
```

### 4. Directory Structure

```
project_root/
├── .env                    # NOT in git, local secrets
├── .env.example            # Template, safe to commit
├── config/
│   ├── backends.yaml       # Backend configs (no secrets)
│   ├── secrets.yaml.example
│   └── credentials/        # NOT in git
│       ├── ssh/
│       │   ├── id_rsa      # Private keys (600 perms)
│       │   └── id_rsa.pub
│       └── api/
│           └── service-account.json
├── .gitignore              # Ignore all secrets
└── src/
```

**.gitignore**:
```
# Secrets
.env
config/secrets.yaml
config/credentials/
*.key
*.pem
*_key
*_secret
*credentials*.json

# But allow examples
!*.example
!*_template.json
```

## Implementation Design

### 1. Secrets Manager

```python
class SecretsManager:
    """Central secrets management.

    Responsibilities:
    - Load secrets from multiple providers
    - Provide unified interface
    - Audit access
    - Mask secrets in logs
    """

    def __init__(self, providers: List[SecretsProvider]):
        self.providers = providers
        self._audit_log = []

    def get_secret(
        self,
        key: str,
        required: bool = True,
        scope: SecretScope = SecretScope.GLOBAL
    ) -> str | None:
        """Get secret from first provider that has it."""

        # Try each provider in order
        for provider in self.providers:
            value = provider.get_secret(key)
            if value:
                # Audit
                self._audit_log.append({
                    "key": key,
                    "provider": provider.__class__.__name__,
                    "timestamp": datetime.now(),
                    "scope": scope
                })
                return value

        if required:
            raise SecretNotFoundError(f"Required secret '{key}' not found")

        return None

    def mask_secret(self, text: str, secret: str) -> str:
        """Replace secret in text with [REDACTED]."""
        if not secret:
            return text
        return text.replace(secret, "[REDACTED]")
```

### 2. Environment Secrets Provider

```python
class EnvironmentSecretsProvider(SecretsProvider):
    """Load secrets from environment variables.

    Convention: EMBODIED_AI_{KEY_NAME}
    """

    def __init__(self, prefix: str = "EMBODIED_AI_"):
        self.prefix = prefix

    def get_secret(self, key: str) -> str | None:
        env_key = f"{self.prefix}{key.upper()}"
        return os.getenv(env_key)

    def list_keys(self) -> List[str]:
        prefix_len = len(self.prefix)
        return [
            k[prefix_len:].lower()
            for k in os.environ
            if k.startswith(self.prefix)
        ]
```

### 3. File Secrets Provider

```python
class FileSecretsProvider(SecretsProvider):
    """Load secrets from files with proper permissions.

    Security checks:
    - File must have 600 or 400 permissions (not world-readable)
    - Parent directory must be owned by current user
    - Symlinks not followed by default
    """

    def __init__(self, secrets_dir: Path):
        self.secrets_dir = Path(secrets_dir)
        self._validate_directory()

    def _validate_directory(self):
        """Ensure secrets directory has correct permissions."""
        if not self.secrets_dir.exists():
            raise SecretError(f"Secrets directory not found: {self.secrets_dir}")

        # Check ownership
        stat = self.secrets_dir.stat()
        if stat.st_uid != os.getuid():
            raise SecretError("Secrets directory not owned by current user")

        # Check permissions (should be 700 or 750)
        mode = oct(stat.st_mode)[-3:]
        if int(mode[1:]) > 0:  # Others have any permission
            raise SecretError(f"Insecure permissions on {self.secrets_dir}: {mode}")

    def get_secret(self, key: str) -> str | None:
        # Map key to file path
        file_path = self.secrets_dir / key.replace("_", "/")

        if not file_path.exists():
            return None

        # Security check: permissions
        stat = file_path.stat()
        mode = oct(stat.st_mode)[-3:]

        if mode not in ["400", "600"]:
            raise SecretError(
                f"Insecure file permissions {mode} for {file_path}. "
                f"Should be 400 or 600."
            )

        return file_path.read_text().strip()
```

### 4. Vault Secrets Provider

```python
class VaultSecretsProvider(SecretsProvider):
    """HashiCorp Vault integration.

    Uses AppRole authentication for services.
    """

    def __init__(self, vault_addr: str, role_id: str, secret_id: str):
        import hvac  # Optional dependency

        self.client = hvac.Client(url=vault_addr)

        # Authenticate
        self.client.auth.approle.login(
            role_id=role_id,
            secret_id=secret_id
        )

    def get_secret(self, key: str) -> str | None:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=key
            )
            return response["data"]["data"]["value"]
        except Exception:
            return None
```

### 5. Backend Configuration

```yaml
# config/backends.yaml (safe to commit)
backends:
  ssh_remote_gpu_1:
    type: ssh
    host: gpu-server-1.example.com
    port: 22
    user: benchmark
    # No secrets here!
    # References to secrets only
    ssh_key: "${secret:ssh_remote_gpu_1_private_key}"
    # Or use environment variable reference
    # ssh_key: "${env:EMBODIED_AI_SSH_KEY_GPU1}"

  kubernetes_cluster:
    type: kubernetes
    cluster_name: ml-cluster
    namespace: benchmarks
    kubeconfig: "${secret:k8s_kubeconfig}"

  aws_ec2:
    type: aws
    region: us-west-2
    instance_type: g4dn.xlarge
    credentials: "${secret:aws_credentials}"
```

### 6. Secret Rotation

```python
class SecretRotation:
    """Automatic secret rotation."""

    def __init__(self, secrets_manager: SecretsManager):
        self.manager = secrets_manager
        self.rotation_policies = {}

    def register_policy(
        self,
        key: str,
        max_age_days: int,
        rotation_func: Callable
    ):
        """Register rotation policy for a secret."""
        self.rotation_policies[key] = {
            "max_age": timedelta(days=max_age_days),
            "rotator": rotation_func
        }

    def check_rotation_needed(self, key: str) -> bool:
        """Check if secret needs rotation."""
        policy = self.rotation_policies.get(key)
        if not policy:
            return False

        last_rotation = self.manager.get_secret_metadata(key).get("last_rotated")
        if not last_rotation:
            return True

        age = datetime.now() - last_rotation
        return age > policy["max_age"]
```

## Usage Patterns

### Development (Local)

```bash
# 1. Copy example
cp .env.example .env

# 2. Add your secrets
echo "EMBODIED_AI_SSH_KEY_DEV=$HOME/.ssh/id_rsa" >> .env
echo "EMBODIED_AI_SSH_HOST_DEV=my-gpu-server.local" >> .env

# 3. Run workflow (secrets loaded automatically)
python examples/simple_workflow.py
```

### Production (Vault)

```python
# Initialize with Vault provider
secrets_manager = SecretsManager([
    VaultSecretsProvider(
        vault_addr=os.getenv("VAULT_ADDR"),
        role_id=os.getenv("VAULT_ROLE_ID"),
        secret_id=os.getenv("VAULT_SECRET_ID")
    ),
    EnvironmentSecretsProvider()  # Fallback
])

# Backend automatically uses secrets
backend = RemoteSSHBackend(
    host="gpu-server.example.com",
    secrets_manager=secrets_manager
)
```

### CI/CD

```yaml
# .github/workflows/benchmark.yml
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Configure secrets
        run: |
          mkdir -p config/credentials/ssh
          echo "${{ secrets.SSH_PRIVATE_KEY }}" > config/credentials/ssh/id_rsa
          chmod 600 config/credentials/ssh/id_rsa

      - name: Run benchmark
        env:
          EMBODIED_AI_SSH_HOST: ${{ secrets.SSH_HOST }}
          EMBODIED_AI_SSH_USER: ${{ secrets.SSH_USER }}
        run: python run_benchmark.py
```

## Logging and Auditing

### Secure Logging

```python
class SecureLogger:
    """Logger that automatically redacts secrets."""

    def __init__(self, logger, secrets_manager: SecretsManager):
        self.logger = logger
        self.secrets = secrets_manager

    def info(self, message: str, **kwargs):
        # Mask all known secrets
        safe_message = self.secrets.mask_all_secrets(message)
        safe_kwargs = {
            k: self.secrets.mask_all_secrets(str(v))
            for k, v in kwargs.items()
        }
        self.logger.info(safe_message, **safe_kwargs)
```

### Audit Trail

```json
{
  "timestamp": "2025-11-02T14:30:00Z",
  "action": "secret_accessed",
  "key": "ssh_remote_gpu_1_private_key",
  "provider": "FileSecretsProvider",
  "user": "john.doe",
  "ip": "10.0.1.42",
  "success": true,
  "backend": "RemoteSSHBackend"
}
```

## Best Practices

### DO ✅

1. **Use environment variables for dev**
```bash
export EMBODIED_AI_SSH_KEY=$HOME/.ssh/id_rsa
```

2. **Use secret management services in production**
```python
secrets = VaultSecretsProvider(...)
```

3. **Check file permissions**
```bash
chmod 600 config/credentials/ssh/id_rsa
```

4. **Use .env files (not committed)**
```bash
echo ".env" >> .gitignore
```

5. **Provide .example templates**
```bash
# .env.example
EMBODIED_AI_SSH_HOST=your-server.example.com
EMBODIED_AI_SSH_USER=benchmark
EMBODIED_AI_SSH_KEY=/path/to/key
```

6. **Rotate secrets regularly**
```python
rotation.register_policy("api_token", max_age_days=90, ...)
```

7. **Audit access**
```python
secrets_manager.get_audit_log()
```

### DON'T ❌

1. **Never hardcode secrets**
```python
# BAD!
ssh_key = "/home/user/.ssh/id_rsa"
api_token = "sk-1234567890abcdef"
```

2. **Never commit secrets**
```bash
# BAD! Never do this
git add config/credentials/
```

3. **Never log secrets**
```python
# BAD!
logger.info(f"Using API key: {api_key}")

# GOOD
logger.info("Using API key: [REDACTED]")
```

4. **Never use weak permissions**
```bash
# BAD!
chmod 644 id_rsa  # World-readable!

# GOOD
chmod 600 id_rsa  # Owner only
```

5. **Never share secrets via insecure channels**
- Slack, email, SMS → NO
- Encrypted channels, vault → YES

## Security Checklist

- [ ] .gitignore configured to exclude secrets
- [ ] .env.example provided as template
- [ ] File permissions validated (600 for keys)
- [ ] Secrets loaded from secure sources
- [ ] Audit logging enabled
- [ ] Secrets masked in logs and reports
- [ ] Least privilege: backends only get needed secrets
- [ ] Rotation policies defined
- [ ] Error messages don't expose secrets
- [ ] Integration tests use fake/mock secrets
- [ ] Production uses vault service
- [ ] CI/CD secrets stored in platform's secret manager
- [ ] Documentation includes security guidelines
- [ ] Code review includes security check

## Future Enhancements

1. **Secret Encryption at Rest**
   - Encrypt local secret files
   - Use OS keychain for master key

2. **Short-lived Credentials**
   - Generate temporary SSH keys
   - Use IAM roles instead of static keys

3. **Just-in-Time Access**
   - Request access when needed
   - Automatic revocation after use

4. **Anomaly Detection**
   - Alert on unusual access patterns
   - Rate limiting

5. **Secret Scanning**
   - Pre-commit hooks to detect secrets
   - Automated scanning of codebase
