import torch
import unittest
from src.models.configuration_guard import EthicalGuardConfig
from src.models.modeling_guard import EthicalGuardModel

class TestEthicalGuardArchitecture(unittest.TestCase):
    def setUp(self):
        # Scale parameters down aggressively for ultra-fast compilation unit testing
        self.config = EthicalGuardConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=1,
            intermediate_size=512,
            max_position_embeddings=512
        )
        self.model = EthicalGuardModel(self.config)

    def test_forward_pass_shapes(self):
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        logits, loss = self.model(input_ids=input_ids, labels=input_ids)
        
        # Logits must match [Batch, Sequence, Vocab]
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        # Loss must compile down to a scalar value
        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)

if __name__ == "__main__":
    unittest.main()