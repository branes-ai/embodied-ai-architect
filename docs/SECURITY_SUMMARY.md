# Security Architecture Summary

## Overview

The Embodied AI Architect implements a comprehensive security system for managing credentials and secrets required for remote benchmarking. The architecture follows industry best practices and demonstrates clean separation of concerns.

## Design Principles

### 1. Never Hardcode Secrets ✅
```python
# ❌ BAD - Never do this!
ssh_key = "/home/user/.ssh/id_rsa"
api_token = "sk-1234567890abcdef"

# ✅ GOOD - Use secrets manager
ssh_key = secrets_manager.get_secret("ssh_key")
```

### 2. Defense in Depth ✅
Multiple layers of security:
- Secret providers (environment, files, vault)
- File permission validation (600/400 only)
- Directory ownership checks
- Automatic secret masking
- Audit logging

### 3. Fail Secure ✅
```python
# Errors deny access, never grant it
secret = secrets_manager.get_secret("key", required=True)
# Raises SecretNotFoundError if not found - never returns None
```

### 4. Separation of Concerns ✅
- **Secrets Layer**: Handles credential management
- **Backend Layer**: Uses secrets, doesn't manage them
- **Orchestrator**: Coordinates workflow, unaware of secrets

## Architecture Components

### 1. SecretsManager (Core)

```
┌─────────────────────────────────────────────────────────┐
│  SecretsManager                                         │
│  - Try providers in order                               │
│  - Cache secrets                                        │
│  - Audit all accesses                                   │
│  - Mask secrets in logs                                 │
└─────────────────────────────────────────────────────────┘
                           ↓
            ┌──────────────┴──────────────┐
            ↓                             ↓
┌─────────────────────────┐  ┌──────────────────────────┐
│ EnvironmentSecrets      │  │ FileSecretsProvider      │
│ Provider                │  │                          │
│ - EMBODIED_AI_* vars    │  │ - config/credentials/    │
│ - Development use       │  │ - Permission checks      │
└─────────────────────────┘  └──────────────────────────┘
```

**Key Features**:
- **Multi-Provider**: Try environment, then files, then vault
- **Audit Logging**: Every access is logged
- **Secret Masking**: Automatic redaction in logs/errors
- **Reference Resolution**: `${secret:key}` and `${env:VAR}`

### 2. Secrets Providers

#### EnvironmentSecretsProvider
```bash
# Development use
export EMBODIED_AI_SSH_HOST=gpu-server.example.com
export EMBODIED_AI_SSH_USER=benchmark
export EMBODIED_AI_SSH_KEY=/path/to/key
```

#### FileSecretsProvider
```
config/credentials/
├── ssh/
│   ├── remote_key      # Mode: 600 (owner read/write only)
│   └── id_rsa          # Mode: 600
└── api/
    └── token           # Mode: 600
```

**Security Checks**:
- Files must have 600 or 400 permissions
- Directory must be owned by current user
- Directory should have 700 permissions
- Fails fast if insecure

### 3. RemoteSSHBackend

```
┌─────────────────────────────────────────────────────────┐
│  RemoteSSHBackend                                       │
│  1. Get SSH key from SecretsManager (never hardcoded)   │
│  2. Connect via paramiko with key auth                  │
│  3. Transfer model via SFTP                             │
│  4. Execute benchmark script remotely                   │
│  5. Retrieve results                                    │
│  6. Cleanup remote files                                │
│  7. Close connection                                    │
└─────────────────────────────────────────────────────────┘
```

**Security Features**:
- Secrets obtained from SecretsManager
- No password authentication (key-based only)
- Automatic cleanup on completion/error
- Connection pooling and reuse
- Proper error masking

## Usage Patterns

### Development (Local)

```bash
# 1. Copy example
cp .env.example .env

# 2. Configure secrets
nano .env

# 3. Set permissions
chmod 600 .env

# 4. Run
python examples/remote_benchmark_example.py
```

### Production (Vault)

```python
from embodied_ai_architect.security import SecretsManager, VaultSecretsProvider

secrets = SecretsManager([
    VaultSecretsProvider(
        vault_addr=os.getenv("VAULT_ADDR"),
        role_id=os.getenv("VAULT_ROLE_ID"),
        secret_id=os.getenv("VAULT_SECRET_ID")
    )
])

backend = RemoteSSHBackend(
    host="gpu-cluster.example.com",
    user="benchmark",
    secrets_manager=secrets
)
```

### CI/CD

```yaml
# GitHub Actions
jobs:
  benchmark:
    steps:
      - name: Configure SSH
        run: |
          mkdir -p config/credentials/ssh
          echo "${{ secrets.SSH_KEY }}" > config/credentials/ssh/id_rsa
          chmod 600 config/credentials/ssh/id_rsa

      - name: Run benchmark
        env:
          EMBODIED_AI_SSH_HOST: ${{ secrets.SSH_HOST }}
          EMBODIED_AI_SSH_USER: ${{ secrets.SSH_USER }}
        run: python run_benchmark.py
```

## Security Features in Action

### 1. Secret Masking

```python
# Before
print(f"Connecting with key: {ssh_key}")
# Output: Connecting with key: /home/user/.ssh/id_rsa

# After (with secrets_manager)
message = f"Connecting with key: {ssh_key}"
safe_message = secrets_manager.mask_secret(message, ssh_key)
print(safe_message)
# Output: Connecting with key: [REDACTED]
```

### 2. Audit Logging

```python
{
  "timestamp": "2025-11-02T14:30:00Z",
  "action": "secret_accessed",
  "key": "ssh_key",
  "provider": "EnvironmentSecretsProvider",
  "success": true
}
```

### 3. Configuration References

```yaml
# backends.yaml
backends:
  remote_gpu:
    host: "${env:EMBODIED_AI_SSH_HOST}"
    user: "${env:EMBODIED_AI_SSH_USER}"
    key: "${secret:ssh_key}"
```

Resolved at runtime:
```python
config = yaml.load("backends.yaml")
resolved = secrets_manager.resolve_references(config)
# All ${...} references are replaced with actual values
```

### 4. File Permission Validation

```python
# Automatically checks file permissions
try:
    provider = FileSecretsProvider("config/credentials")
except SecretError as e:
    # Error: Insecure permissions 644 for /config/credentials/ssh/key
    # Should be 400 or 600
    print(f"Security error: {e}")
```

## Protection Against Common Threats

### 1. Accidental Git Commits ✅
```gitignore
# .gitignore
.env
config/credentials/
*.key
*.pem
*_secret
*credentials*.json
```

### 2. Log Exposure ✅
```python
# All logs automatically masked
logger.info(f"Using key: {ssh_key}")
# Actually logs: "Using key: [REDACTED]"
```

### 3. Error Message Leakage ✅
```python
# Exceptions automatically mask secrets
try:
    connect(key=ssh_key)
except Exception as e:
    # Exception message: "Connection failed: [REDACTED]"
    print(e)  # Secret is masked
```

### 4. Insecure File Permissions ✅
```python
# Automatically detected and rejected
# Files must be 600 or 400 (owner only)
if mode not in ["400", "600"]:
    raise SecretError("Insecure file permissions")
```

## Best Practices Checklist

Development:
- ✅ Use .env file (not committed)
- ✅ Copy from .env.example
- ✅ Set file permissions: `chmod 600`
- ✅ Use environment variables

Production:
- ✅ Use vault service (HashiCorp Vault, AWS Secrets Manager)
- ✅ Use short-lived credentials
- ✅ Rotate secrets regularly
- ✅ Audit access logs

All Environments:
- ✅ Never hardcode secrets
- ✅ Never commit secrets to git
- ✅ Never log secret values
- ✅ Use least privilege
- ✅ Enable audit logging

## Security Boundaries

### What This System Protects Against ✅
- Accidental secret exposure in code
- Secrets in version control
- Secrets in logs and reports
- Insecure file permissions
- Unauthorized secret access

### What This System Does NOT Protect Against ⚠️
- Memory scraping attacks
- Compromised vault services
- Physical access to machines
- Zero-day exploits
- Side-channel attacks

## Future Enhancements

1. **Secret Rotation**
   ```python
   rotation.register_policy("api_token", max_age_days=90, rotator=rotate_token)
   ```

2. **Short-lived Credentials**
   - Generate temporary SSH keys
   - Use IAM roles instead of static keys
   - Auto-revoke after use

3. **Just-in-Time Access**
   - Request access when needed
   - Automatic revocation after timeout

4. **Anomaly Detection**
   - Alert on unusual access patterns
   - Rate limiting
   - Geographic checks

## Testing Security

```python
# Test secret masking
def test_secret_masking():
    secrets = SecretsManager([])
    text = "My key is abc123"
    masked = secrets.mask_secret(text, "abc123")
    assert "abc123" not in masked
    assert "[REDACTED]" in masked

# Test file permissions
def test_file_permissions():
    with pytest.raises(SecretError):
        # Should fail with insecure permissions
        FileSecretsProvider("/path/with/644/files")

# Test audit logging
def test_audit_logging():
    secrets = SecretsManager([...])
    secrets.get_secret("test_key")
    audit = secrets.get_audit_log()
    assert len(audit) > 0
    assert audit[0]["key"] == "test_key"
```

## Conclusion

The security architecture provides:
- **Defense in Depth**: Multiple layers of protection
- **Ease of Use**: Simple API, hard to misuse
- **Production Ready**: Supports vault services, audit logging
- **Well Documented**: Clear examples and guidelines
- **Tested**: Security features validated

**Key Takeaway**: Secrets are treated as first-class concerns, not afterthoughts. The system makes it easy to do the right thing and hard to do the wrong thing.
