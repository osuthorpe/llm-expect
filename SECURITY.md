# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them privately:

1. **Email**: security@llm-expect.dev (if available)
2. **GitHub Security Advisories**: Use the "Report a vulnerability" button on our GitHub repository
3. **Response Time**: We aim to respond within 48 hours

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

## Security Considerations

### API Key Management

**❌ Never do this:**
```python
# Don't hardcode API keys
@llm_expect(judge_provider="openai", api_key="sk-...")
def my_function():
    pass

# Don't commit keys to git
export OPENAI_API_KEY="sk-proj-..."
```

**✅ Best practices:**
```python
# Use environment variables
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Use key management services in production
# - AWS Secrets Manager
# - Azure Key Vault
# - HashiCorp Vault
```

### Dataset Security

**🛡️ Sensitive Data Protection:**
- **Never include PII** in test datasets
- **Sanitize real data** before using in evaluations
- **Use synthetic data** when possible
- **Encrypt datasets at rest** for production use

**Example - Sanitizing datasets:**
```python
# Bad - real user data
{"id": "user123", "input": "My SSN is 123-45-6789", ...}

# Good - synthetic/sanitized data  
{"id": "test001", "input": "My SSN is XXX-XX-XXXX", ...}
```

### Judge Provider Security

**🔐 LLM-as-Judge Considerations:**
- **Data residency**: Understand where your data is processed
- **Retention policies**: Know how long providers store your data
- **Access controls**: Use API keys with minimal required permissions
- **Audit logging**: Enable logging for compliance requirements

### Network Security

**🌐 API Security:**
- Use **HTTPS only** for all API calls
- Implement **rate limiting** to prevent abuse
- **Validate inputs** before sending to external services
- **Sanitize outputs** from external services

## Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 2**: Initial response and acknowledgment
3. **Day 7**: Vulnerability assessment completed
4. **Day 14**: Fix developed and tested
5. **Day 21**: Security update released
6. **Day 28**: Public disclosure (if appropriate)

## Security Updates

Security updates are released as patch versions (e.g., 0.1.1 → 0.1.2) and are backward compatible whenever possible.

Subscribe to security notifications:
- GitHub repository "Watch" → "Custom" → "Security alerts"
- PyPI project notifications
- Release notes on GitHub

---

**For general questions about security practices, open a public discussion on GitHub.**