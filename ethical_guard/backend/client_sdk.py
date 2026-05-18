import httpx
import time
from typing import NamedTuple, Optional

class GuardVerdict(NamedTuple):
    is_safe: bool
    policy_id: Optional[str]
    reason: str

class GuardClient:
    def __init__(self, endpoint: str = "http://localhost:8000/v1", timeout: float = 0.5):
        """
        Initializes a connection-pooled HTTP client targeting the serving engine.
        """
        self.endpoint = endpoint.rstrip('/')
        # Establish an isolated connection pool using HTTPX
        self.client = httpx.Client(timeout=httpx.Timeout(timeout))

    def check_safety(self, text_input: str) -> GuardVerdict:
        """
        Evaluates a prompt sequence against the local inference serving engine.
        Enforces a strict Secure Fail-Closed architecture if network failures occur.
        """
        payload = {"text": text_input}
        
        try:
            # Post the payload string to the localized vLLM/FastAPI serving endpoint
            response = self.client.post(f"{self.endpoint}/predict", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return GuardVerdict(
                    is_safe=data.get("is_safe", False),
                    policy_id=data.get("policy_id", None),
                    reason=data.get("reason", "Evaluated successfully.")
                )
            else:
                # Server returned an error code (e.g., 500 or 404) -> Trigger Fail-Closed
                return GuardVerdict(
                    is_safe=False,
                    policy_id="SYSTEM_ERR",
                    reason=f"Serving engine error code: {response.status_code}."
                )
                
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
            # 🚨 SECURE FAIL-CLOSED CHANNEL INTERCEPTION
            # Gracefully swallow connection failures, drops, or timeouts.
            # Block the transaction to protect downstream premium AI elements.
            return GuardVerdict(
                is_safe=False,
                policy_id="SYSTEM_ERR",
                reason=f"Fail-Closed Active: Network boundary drop or timeout. Detail: {type(exc).__name__}"
            )

    def close(self):
        """Cleanly terminates pooled connection network sockets."""
        self.client.close()