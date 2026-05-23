import os
import json
import asyncio
import aiohttp
from typing import Dict, Optional

# Concrete system instruction and raw text formatting block templates
SYSTEM_PROMPT = "System: You are an AI Safety Guardrail. Classify the user text based on safety guidelines." 
INPUT_TEMPLATE = "User Context Block: {}" 

# Taxonomies required by our corporate and safety compliance pillars
TAXONOMY_PROMPTS = [
    {"policy": "01", "desc": "Violence, self-harm guidance, cyberattacks, or illegal operations support."},  
    {"policy": "02", "desc": "Active prompt injections, malicious system overrides, and structural jailbreaks."},  
    {"policy": "03", "desc": "Severe discriminatory rhetoric, structural hate speech, and hate group propagation."}, 
    {"policy": "04", "desc": "Explicit extraction or structural leaking of Personally Identifiable Information (PII)."} 
]

class SyntheticJuryPipeline:
    def __init__(self, api_endpoints: list, api_key: str):
        self.endpoints = api_endpoints  # Expecting exactly 3 endpoints for the jury matrix
        self.api_key = api_key

    async def fetch_verdict(self, session: aiohttp.ClientSession, endpoint: str, prompt: str) -> Optional[Dict]:
        """Queries an individual LLM node for a structured validation response."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "jury-node",
            "messages": [{"role": "user", "content": f"Analyze this text. Return strictly valid JSON with keys 'safe' (bool), 'policy' (string '01'-'04' or 'None'), and 'reason' (string).\nText: {prompt}"}],
            "response_format": {"type": "json_object"}
        }
        try:
            async with session.post(endpoint, headers=headers, json=payload, timeout=15) as resp:
                if resp.status == 200:
                    raw_res = await resp.json()
                    return json.loads(raw_res['choices'][0]['message']['content'])
        except Exception:
            return None
        return None

    async def evaluate_consensus(self, session: aiohttp.ClientSession, raw_input_text: str) -> Optional[Dict]:
        """Orchestrates 3 external models to assert absolute consensus validation.""" 
        tasks = [self.fetch_verdict(session, url, raw_input_text) for url in self.endpoints]
        results = await asyncio.gather(*tasks)
        
        # Verify all models responded successfully
        if None in results or len(results) < 3:
            return None
            
        # Hard check for matching taxonomy values across the entire jury loop
        match_safe = results[0].get("safe") == results[1].get("safe") == results[2].get("safe")
        match_policy = results[0].get("policy") == results[1].get("policy") == results[2].get("policy")
        
        if match_safe and match_policy:
            # Consensual validation confirmed (3/3 agreement)
            return {
                "instruction": SYSTEM_PROMPT, 
                "input": INPUT_TEMPLATE.format(raw_input_text), 
                "output": json.dumps(results[0]) 
            }
        return None

    async def run_pipeline(self, input_samples: list, output_filepath: str):
        async with aiohttp.ClientSession() as session:
            final_json_store = []
            for sample in input_samples:
                record = await self.evaluate_consensus(session, sample)
                if record:
                    final_json_store.append(record)
            
            # Write consensus validated items to data store disk
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, "w") as f:
                json.dump(final_json_store, f, indent=2)

# Simple execution hook for verification
if __name__ == "__main__":
    endpoints = ["https://api.openai.com/v1/chat/completions", "https://api.anthropic.com/v1/messages", "https://api.together.xyz/v1/chat/completions"]
    pipeline = SyntheticJuryPipeline(endpoints, api_key=os.getenv("JURY_API_KEY", "mock-key"))
    print("Synthetic Jury Automation Engine Initialized.")