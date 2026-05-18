import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel
from src.models.configuration_guard import EthicalGuardConfig

class RoPEEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len, :].to(x.device), self.sin_cached[:seq_len, :].to(x.device)

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Shapes: [B, H, S, D_head]
        cos = cos.unsqueeze(0).unsqueeze(1) 
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: EthicalGuardConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: EthicalGuardConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_queries_per_kv = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, rope: RoPEEmbedding, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, C = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rope(q, S)
        q = rope.apply_rope(q, cos, sin)
        k = rope.apply_rope(k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, C)
        return self.out_proj(attn_output)

class CausalDecoderLayer(nn.Module):
    def __init__(self, config: EthicalGuardConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor, rope: RoPEEmbedding, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class EthicalGuardModel(PreTrainedModel):
    config_class = EthicalGuardConfig

    def __init__(self, config: EthicalGuardConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RoPEEmbedding(dim=config.hidden_size // config.num_heads, max_position_embeddings=config.max_position_embeddings)
        
        self.layers = nn.ModuleList([CausalDecoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, S = input_ids.shape
        x = self.embed(input_ids)

        mask = torch.full((S, S), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, self.rope, mask)
        
        logits = self.lm_head(self.final_norm(x))
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss
