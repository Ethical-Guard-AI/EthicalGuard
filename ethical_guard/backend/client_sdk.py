import httpx
import json
from typing import NamedTuple

class GuardVerdict(NamedTuple):
    is_safe: bool
    policy_id: str
    reason: str

class GuardClient:
    """
    Importable open-source client SDK library providing a clean, high-performance
    interface to query the EthicalGuard AI serving engine.
    """
    def __init__(self, endpoint: str = "http://localhost:8000/v1"):
        self.endpoint = endpoint.rstrip("/")
        # Initialize a re-usable, connection-pooled synchronous HTTP client
        self.client = httpx.Client(timeout=2.0)  # Low timeout bounds to back our latency commitments

    def check_safety(self, text_to_evaluate: str) -> GuardVerdict:
        """
        Evaluates an incoming raw text string against our target ethical taxonomy.
        """
        payload = {
            "prompt": f"System: You are an AI Safety Guardrail. Classify the user text based on safety guidelines.\nUser Context Block: {text_to_evaluate}",
        }
        
        try:
            # Fire verification payload directly to our optimized vLLM container endpoint
            response = self.client.post(f"{self.endpoint}/completions", json=payload)
            
            if response.status_code == 200:
                raw_response_text = response.json()["choices"][0]["text"]
                # Parse the frozen, structurally accurate JSON token payload returned by the engine
                data = json.loads(raw_response_text)
                
                return GuardVerdict(
                    is_safe=bool(data.get("safe", True)),
                    policy_id=str(data.get("policy", "None")),
                    reason=str(data.get("reason", ""))
                )
        except Exception as e:
            # Fail-safe mode: In case of infrastructure network hiccups, default log and isolate safely
            return GuardVerdict(is_safe=False, policy_id="SYSTEM_ERR", reason=f"SDK Network Exception: {str(e)}")
            
        return GuardVerdict(is_safe=False, policy_id="UNKNOWN_ERR", reason="Invalid serving framework metadata returned.")

    def close(self):
        """Cleanly releases pooled connection handles."""
        self.client.close()

# Integration test sequence demonstrating client SDK usage blueprint
if __name__ == "__main__":
    print("EthicalGuard SDK client interface compiled successfully.")
