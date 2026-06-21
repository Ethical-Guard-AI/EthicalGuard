# Contributing to EthicalGuard AI

Thank you for your interest in contributing to EthicalGuard AI!

**EthicalGuard is open source** — the code, license, and documentation are public and free to use, fork, and audit. However, **direct code contributions are by invitation only**. This is a deliberate choice, not an oversight (see [Why invite-only?](https://www.google.com/search?q=%23why-invite-only) below).

This document explains how to request access, what happens once you have it, and how to report security issues (which follow a separate process regardless of contributor status).

---

## If You Are Not Yet an Invited Collaborator

> ⚠️ **Important:** Please do not open pull requests if you are not an invited collaborator. Unsolicited PRs against this repository will be closed without review, regardless of quality. This isn't a judgment on the contribution; it is a consequence of our strict access model.

### How to Request Access

If you would like to contribute, please email the admin team at **ethicalguardai@gmail.com** with the following information:

1. **A short description** of what you'd like to work on (a specific bug, feature, or area of the codebase — not just *"I want to contribute"*).
2. **Relevant background** (e.g., ML engineering, security research, technical writing) if it relates to your proposal.
3. **A link** to your GitHub profile.

**Please wait for a response before doing any work.** This protects your time — if the area you're interested in is already being worked on internally, or doesn't fit current priorities, you'll find out before investing effort.

If accepted, you'll be added as a collaborator and given the workflow outlined in the Once You're an Invited Collaborator section below.

### What You Can Always Do (No Invitation Required)

* **Open an issue** to report a bug or suggest something, without requesting collaborator access.
* **Fork the repository** under the terms of the MIT License for your own use.
* **Use GitHub Discussions** to ask questions.

---

## Why Invite-Only?

EthicalGuard is a safety-classification system. Its core value relies on the verdict logic (`ethical_guard/models/`, the training pipeline, and the policy taxonomy) being correct and completely uncompromised.

An unreviewed or loosely vetted change to that logic is a **security-relevant change**, not just a code-quality one — similar to how you wouldn't want arbitrary external PRs landing in an authentication library without prior vetting.

Restricting direct contribution to a known, invited set of collaborators lets the maintainers keep the review bar high without having to triage a public PR queue for every change. It does not restrict who can use, audit, or fork the project — the source remains fully open under the MIT License.

> 💡 **Note:** This policy may change as the project matures (e.g., opening up `docs/`-only contributions, or moving to a public PR model with required reviews). Any changes will be reflected here first.

---

## Reporting Security Issues

Security vulnerabilities — including classification bypasses, jailbreaks producing false negatives, DoS vectors, or SDK auth bypasses — **should not** be reported via email or GitHub Issues.

Regardless of your contributor status, please follow the private disclosure process detailed in **`SECURITY.md`**.

---

## Once You're an Invited Collaborator

The guidelines below apply once you have been officially added to the repository with write access.

### Code of Conduct

Be respectful, assume good faith, and keep disagreements focused on the technical substance. Maintainers reserve the right to close or lock issues/PRs that become personal or unproductive.

### Before You Start on a Change

* **Check open issues and PRs** to avoid duplicate work.
* **Open an issue before making a large change.** For anything beyond a small fix — such as new model behavior, a new policy category, a change to the SDK's public interface, or a new training script — open an issue first to discuss the approach.

### Project Layout

Please place any new modules under the existing package structure rather than introducing a new top-level directory unless previously discussed.

```text
Ethical-Guard/
├── configs/              # Model configuration & hyperparameter cards
├── docs/                 # MkDocs documentation source
├── ethical_guard/        # Core package
│   ├── models/           # Attention/layer definitions (GQA, RoPE, etc.)
│   ├── training/         # SFT trainer, data collators
│   │   └── trainer_sft.py
│   └── backend/          # Serving engine + client SDK
│       └── client_sdk.py
├── tests/                # unittest-based test suite
│   ├── test_model_layers.py
│   ├── test_sft_masking.py
│   ├── test_sft_hardening.py
│   └── test_benchmark_latency.py
├── mkdocs.yml
├── pyproject.toml
└── SECURITY.md

```

### Local Development Setup

* **Requirement:** Python ≥ 3.10 (per `pyproject.toml`).

```bash
# 1. Clone the repository directly
git clone https://github.com/Ethical-Guard-AI/EthicalGuard.git
cd EthicalGuard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dependencies
pip install -e .

# 4. Set the Python path to the repo root (needed for test discovery)
export PYTHONPATH=.              # Windows PowerShell: $env:PYTHONPATH="."
```

> 📦 **Dependencies:** Core runtime dependencies are `httpx`, `torch`, and `transformers`. If your change adds a new dependency, justify it in your PR description. This project intentionally avoids heavy dependencies in favor of local, self-contained inference.

### Making Your Change

1. **Branch from main** using a descriptive prefix: `fix/`, `feat/`, `docs/`, `test/`, or `refactor/`.
```bash
git checkout -b fix/short-description
```

2. **Keep changes scoped.** Aim for one logical change per PR. A bug fix shouldn't also reformat unrelated files.
3. **Match the existing style.** Follow the patterns already established in `ethical_guard/` (e.g., type hints on public functions, docstrings on classes, and the structured-tuple style for SDK return values: `is_safe, policy_id, reason`).
4. **Explain safety impact.** If you touch the safety-classification path (`models/`, `training/`, the policy taxonomy, or the verdict schema), explain **why** in your PR description, not just *what*. These changes affect production deployments; reviewers will require justification and test coverage proving no regressions on the adversarial dataset.


### Testing
Run the full suite from the repository root before opening a PR:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

All tests must report `ok`. The test suite includes:

* `test_model_layers.py` — Verifies tensor and layer shapes.
* `test_sft_masking.py` — Verifies label-masking tensors produced by the SFT data collator (prompt tokens masked to `-100`, response tokens left intact). **Double-check this specifically if you modify the token-masking collator;** it is sensitive to silent breakage.
* `test_sft_hardening.py` — Adversarial and contrastive dataset checks.
* `test_benchmark_latency.py` — Enforces the sub-500ms latency ceiling.

If you add new behavior, **add a corresponding test file** matching the `test_*.py` pattern under `tests/`. It will be automatically picked up by test discovery.

### Documentation Changes

Documentation is built with MkDocs (Material theme). Update the files accordingly based on your changes:

* **SDK Public Interface** (`GuardClient`, `check_safety`, verdict shapes) → Update `docs/user_manual.md`
* **Internals / Pipelines** (Repo structure, training pipeline) → Update `docs/developer_manual.md`
* **Security Posture** (Supported versions, disclosure process) → Update `docs/security.md` and/or `SECURITY.md`

Preview your documentation changes locally:

```bash
pip install mkdocs-material
mkdocs serve
```

*Note: Do not add performance numbers or accuracy claims to the documentation unless they come from a reproducible source included in the PR (script, dataset reference, or CI artifact).*

### Commit Messages

Write commit messages that explain *why*, not just *what*:

```text
fix(sdk): close httpx client on connection timeout

Previously a timeout left the connection pool open, leaking
file descriptors under sustained retry load.

```

Please **squash** trivial "fix typo" or "address review" commits before merging if your branch has accumulated several.

### Submitting a Pull Request

1. Push your branch and open a PR against `main`.
2. Fill out the template explaining what changed and why. Link any related issues.
3. Confirm in the description that `python3 -m unittest discover -s tests -v` passes locally.
4. Be responsive to review comments. If you disagree with a requested change, explain your reasoning constructively rather than re-pushing silently.

Review may take a few days. If you haven't heard back after a week, feel free to ping the PR (please do not open a duplicate).

---

## License

By contributing, you agree that your contributions will be licensed under the project's **MIT License**.
