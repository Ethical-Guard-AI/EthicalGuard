from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
FAIL_CLOSED_POLICY_ID = "SYSTEM_ERR"


@dataclass(frozen=True)
class GuardVerdict:
    is_safe: bool
    policy_id: str
    reason: str
    confidence: float = 0.0


Verdict = GuardVerdict


class GuardClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.Client(base_url=self._endpoint, timeout=timeout, transport=transport)

    def __enter__(self) -> "GuardClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def check_safety(self, prompt: str) -> GuardVerdict:
        try:
            response = self._client.post("/completions", json={"prompt": prompt})
            response.raise_for_status()
            data = response.json()
            verdict_data = self._extract_verdict_data(data)
            return self._build_verdict(verdict_data)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            return GuardVerdict(
                is_safe=False,
                policy_id=FAIL_CLOSED_POLICY_ID,
                reason=f"SDK request failed closed: {exc}",
                confidence=0.0,
            )

    def _extract_verdict_data(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if "safe" in payload or "policy" in payload or "reason" in payload:
                return payload

            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    if isinstance(first_choice.get("text"), str):
                        parsed = json.loads(first_choice["text"])
                        if isinstance(parsed, dict):
                            return parsed
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                return parsed

        raise ValueError("Unsupported guard response shape")

    def _build_verdict(self, payload: dict[str, Any]) -> GuardVerdict:
        is_safe = bool(payload.get("safe", False))
        policy_id = str(payload.get("policy", "None" if is_safe else FAIL_CLOSED_POLICY_ID))
        reason = str(payload.get("reason", ""))
        confidence_value = payload.get("confidence")
        confidence = float(confidence_value) if confidence_value is not None else (1.0 if is_safe else 0.0)
        return GuardVerdict(
            is_safe=is_safe,
            policy_id=policy_id,
            reason=reason,
            confidence=confidence,
        )


__all__ = ["GuardClient", "GuardVerdict", "Verdict"]
