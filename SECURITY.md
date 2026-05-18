# 🛡️ Security Policy

## 📋 Operational Security Philosophy

EthicalGuard AI operates on a **Zero-Trust, Fail-Closed** engineering model. Because our framework serves as an inline interceptor for production Large Language Model (LLM) pipelines, security vulnerabilities within the guardrail engine itself represent critical structural risks to downstream applications. 

This document outlines our supported versions, reporting protocols, and baseline security configurations to ensure containment of adversarial attacks.

---

## 🚀 Supported Versions

We actively provide security patches, alignment updates, and vulnerability remediations for the following release tracks:

| Version | Supported | Notes / Remediation Target |
| :--- | :---: | :--- |
| v1.0.x |  Yes | Active Baseline. Main target for adversarial hardening patches. |
| < v1.0.0 | ❌ No | Experimental Alpha Pre-releases. Upgrade immediately to v1.0.0. |

---

## 🔍 Vulnerability Reporting Protocol

**Please do not report security vulnerabilities publicly via GitHub Issues.** If you discover a structural bypass, a novel cognitive alignment exploit ("good in a bad" jailbreak format), a denial-of-service (DoS) vector in our token-freezing grammar engine, or an authentication bypass in the Client SDK, please report it via our private triage channel:

* **Primary Email:** praveenram.ramasubramani@gmail.com / rpstram@gmail.com
* **Target Triage Window:** Validated security reports will receive an initial response within **24 hours**, followed by a comprehensive risk assessment map and remediation timeline within **72 hours**.

### 📝 What to Include in Your Report
To accelerate the validation process, please format your submission with:
1. **Attack Vector Classification:** Identify the target taxonomy pillar breached (Category 01–04).
2. **Reproducible Payload:** Provide the exact raw string or adversarial context block used to break the validation pass.
3. **System Environment Specs:** Detail your runtime environment (e.g., Native Linux, WSL2, Python execution version, virtual environment dependencies).
4. **Output Logs:** Attach the unexpected `GuardVerdict` output showing the false negative response (`is_safe: true`).

---

## 🔒 Baseline Architectural Security Requirements

When deploying EthicalGuard AI inside enterprise networks, developers must adhere to the following infrastructure boundaries:

### 1. Isolated Local Networking (Stateless Execution)
The serving engine (`server_vllm.py`) must be run behind a secure corporate firewall or inside an isolated private network layer (such as a Virtual Private Cloud or Tailscale Mesh Network). 
* **Never expose port `8000` or `0.0.0.0` directly to the public internet** without an authenticating reverse proxy (e.g., NGINX, API Gateways) or mutual TLS (mTLS) decryption loops in place.

### 2. Dependency Audit Enforcement
Because EthicalGuard AI integrates directly with deep-learning frameworks (`torch`, `transformers`, `httpx`), production deployments must be checked regularly for supply-chain vulnerabilities. 
* Run automated safety scans against your active virtual environment parameters:
  ```bash
  pip install safety
  safety check
