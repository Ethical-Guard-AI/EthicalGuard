import json
from typing import AsyncGenerator
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# Our mandatory output format specified in Section 2.2
TARGET_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "safe": {"type": "boolean"},
        "policy": {"type": "string", "enum": ["01", "02", "03", "04", "None"]},
        "reason": {"type": "string"}
    },
    "required": ["safe", "policy", "reason"]
}

class EthicalGuardServingEngine:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        # Configure vLLM core engine primitives
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate_guard_verdict(self, prompt: str, request_id: str) -> AsyncGenerator[str, None]:
        """
        Pushes a text block into the model while enforcing structural JSON output logic.
        """
        # Freeze tokens strictly to JSON formatting rules via guided_json parameter
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic validation execution
            max_tokens=128,
            guided_json=json.dumps(TARGET_JSON_SCHEMA)
        )
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        async for request_output in results_generator:
            # Yield token streams as they clear the forced grammar matrix
            yield request_output.outputs[0].text

if __name__ == "__main__":
    print("vLLM Guardrail Serving Container Scaffolding Successfully Initialized.")
