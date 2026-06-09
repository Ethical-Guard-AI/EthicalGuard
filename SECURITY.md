# Security Policy

## Supported Versions

This project currently supports the latest v2 release line and active development branches. If you are running a packaged release, please verify that you are on the most recent published version before reporting an issue.

## Reporting a Vulnerability

If you believe you have found a security issue, report it privately and include:

- A short summary of the issue
- Affected file(s), endpoint(s), or workflow(s)
- Steps to reproduce
- Expected impact and severity
- Any proof-of-concept payloads or logs that help reproduce the problem

Please avoid posting sensitive details in public issue trackers until the issue has been triaged.

## What to Report

Examples of security-relevant issues include:

- Remote code execution, arbitrary file access, or unsafe deserialization
- Authentication, authorization, or access-control bypasses
- Prompt injection or jailbreak paths that bypass the SDK or server-side safety checks
- Data exposure, secrets leakage, or unexpected logging of user prompts
- Denial-of-service conditions caused by malformed requests or unbounded resource use

## What to Include for This Project

For this repository, please include the exact client or server path involved, such as:

- `ethical_guard/backend/client_sdk.py`
- `ethical_guard/backend/server_vllm.py`
- `runserver.py`
- Any custom endpoint configuration you used

If the issue affects fail-closed behavior, include the observed `policy_id`, `is_safe` value, and the response payload if available.

## Response Expectations

Security reports should receive an acknowledgement after triage. If the issue is confirmed, the maintainers will work with you on a safe fix and disclosure timeline.

## Safe Disclosure Guidance

If possible, delay public disclosure until a patch is available or the maintainers confirm that the issue has been resolved.
