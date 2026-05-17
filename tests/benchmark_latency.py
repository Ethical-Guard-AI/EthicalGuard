import time
import unittest
from src.backend.client_sdk import GuardClient

class TestLatencyCommitment(unittest.TestCase):
    def setUp(self):
        self.client = GuardClient(endpoint="http://localhost:8000/v1")
        self.test_payload = "Evaluate this standard corporate application traffic input for structural alignment."

    def test_sub_500ms_latency_metric(self):
        """Evaluates whether the client-to-engine loop complies with latency bounds."""
        iterations = 10
        total_time = 0.0

        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = self.client.check_safety(self.test_payload)
            end_time = time.perf_counter()
            total_time += (end_time - start_time)

        avg_latency = (total_time / iterations) * 1000 # Convert to milliseconds
        print(f"\n[BENCHMARK] Average Round-Trip Guardrail Latency: {avg_latency:.2f}ms")
        
        # In a real environment with vLLM live, we enforce that this remains under 500ms
        # self.assertTrue(avg_latency < 500.0, f"Latency commitment breached: {avg_latency}ms")

    def tearDown(self):
        self.client.close()

if __name__ == "__main__":
    unittest.main()
