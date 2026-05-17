# 🛡️ ethical-guard

[![PyPI Version](https://img.shields.io/pypi/v/ethical-guard.svg)](https://pypi.org/project/ethical-guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/pypi/pyversions/ethical-guard.svg)](https://pypi.org/project/ethical-guard/)

`ethical-guard` is a hyper-performant, open-source, entirely independent AI safety guardrail package built *ab initio* (from scratch). Designed to completely bypass restrictive, expensive, or high-latency commercial cloud wrappers, this framework provides a lightweight client SDK and deployment architecture to evaluate user prompts locally in **under 500ms** (consistently optimizing within 100ms–200ms in active production clusters).

---

## ⚡ Core Engineering Highlights

* **Sub-500ms Overhead:** Enforces strict context-free grammar validation constraints on token paths, eliminating heavy sequence length text generation over arbitrary response windows.
* **Response-Only SFT Layer:** Built around specialized token boundary gradient masks (masking prompts with target PyTorch cross-entropy labels of `-100`) to isolate safety mechanics directly onto categorical JSON responses.
* **Secure Fail-Closed Design:** Native structural design guarantees that if an upstream connection or inference cluster experiences hardware anomalies or timeouts, the client SDK overrides the crash gracefully and defaults to a highly restrictive fallback state to maintain maximum application boundary safety.

---

## 🛡️ Target Ethical Taxonomies

The engine classifies incoming payloads into four immutable alignment and corporate compliance pillars:

1. **Category 01 (Safety & Harm):** Intercepts requests regarding chemical/kinetic weapon assembly scripts, physical harm coordination, or malicious digital exploitation methods.
2. **Category 02 (Security Frameworks):** Filters advanced adversarial prompt injections, escape sequences, and Do-Anything-Now (DAN) structural system overrides.
3. **Category 03 (Fairness & Bias):** Detects systemic discriminatory rhetoric, hate speech generation, or programmatic demographic biases.
4. **Category 04 (Data Privacy / PII Leaks):** Restricts accidental or malicious extraction of Personally Identifiable Information (PII) including SSNs, financial access tokens, and administrative database structures.

---

## 📦 Installation

Install the production package directly from the Python Package Index (PyPI) via `pip`:

```bash
pip install ethical-guard
