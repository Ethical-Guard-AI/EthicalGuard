import torch
from typing import Dict, List, Any
from transformers import Trainer

class ResponseOnlyDataCollator:
    """
    Custom Data Collator that masks out prompt tokens, enforcing gradient 
    evaluation exclusively on the output JSON object strings.
    """
    def __init__(self, tokenizer: Any, response_template: str = '"output": "'):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.ignore_index = -100  # PyTorch CrossEntropyLoss standard ignore ID

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Standard collation of input_ids and attention_masks
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        
        # Pad sequences to the longest sequence in the current micro-batch
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        # Clone inputs to create initial autoregressive labels
        labels = padded_inputs.clone()
        
        # Encode our response boundary template to track indices
        template_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        template_len = len(template_ids)

        for i in range(len(features)):
            feature_ids = padded_inputs[i].tolist()
            
            # Search for the exact token sequence where the JSON output begins
            match_idx = -1
            for idx in range(len(feature_ids) - template_len):
                if feature_ids[idx : idx + template_len] == template_ids:
                    match_idx = idx + template_len
                    break
            
            if match_idx != -1:
                # Mask out all tokens from the start up to the matching response index
                labels[i, :match_idx] = self.ignore_index
            else:
                # Fallback: If no template match is found, mask out the entire row to prevent corruption
                labels[i, :] = self.ignore_index

        batch["input_ids"] = padded_inputs
        batch["attention_mask"] = padded_masks
        batch["labels"] = labels
        
        return batch

class EthicalGuardSFTTrainer(Trainer):
    """
    Custom Trainer wrapper that leverages our response-only token-masking collator.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force the custom collator setup into the execution loop
        self.data_collator = ResponseOnlyDataCollator(tokenizer=self.tokenizer)
