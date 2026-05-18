from transformers import PretrainedConfig

class EthicalGuardConfig(PretrainedConfig):
    model_type = "ethical_guard"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5632,
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
