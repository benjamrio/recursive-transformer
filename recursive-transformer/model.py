import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect
from einops import rearrange, einsum
import math

@dataclass
class GPTConfig:
  block_size: int = 512
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int = 12
  d_emb: int = 768

class MLP(nn.Module):
  def __init__(self, d_emb):
    super().__init__()
    self.c_fc = nn.Linear(d_emb, 4 * d_emb)
    self.gelu = nn.GELU(approximate="tanh")
    self.c_proj = nn.Linear(4 * d_emb, d_emb)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)


def build_attn_mask(seqlen: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
  return mask

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.d_emb, 3 * config.d_emb)
        self.c_proj = nn.Linear(config.d_emb, config.d_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.d_emb = config.d_emb

    def forward(self, x):
        B, T, d_emb = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_emb, dim=2)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)
        dotproducts = einsum(q, k, 'b h i d, b h j d -> b h i j')
        scale = 1.0 / math.sqrt(k.size(-1)) # root of head dimension
        scores = (dotproducts * scale).to(torch.float32)
        #attn_mask = build_attn_mask(T).to(x.device)
        scores = scores #+ attn_mask
        attention = F.softmax(scores, dim=-1).to(torch.float32)
        output = einsum(attention, v, "b h i j, b h j d -> b h i d")
        output = rearrange(output, 'b h i d -> b i (h d)')
        y = self.c_proj(output)
        return y

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.d_emb)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.d_emb)
    self.mlp = MLP(config.d_emb)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wle = nn.Linear(config.vocab_size, config.d_emb), # logits into embeddings: possible to just take the embedding before conversion in logits
            wte = nn.Embedding(config.vocab_size, config.d_emb),
            wpe = nn.Embedding(config.block_size, config.d_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_emb),
        ))
        self.lm_head = nn.Linear(config.d_emb, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, prst_idxs, past_embs, targets=None):
        B_past, T_past, d_emb = past_embs.size()
        B_prst, T_prst = prst_idxs.size()
        assert (B_past == B_prst)
        assert T_past + T_prst <= self.config.block_size, f"Cannot forward sequence of length {T_past} + {T_prst}, block size is only {self.config.block_size}"

        prst_emb = self.transformer.wte(prst_idxs)
        tok_emb = torch.cat([past_embs, prst_emb], dim=1)

        pos = torch.arange(0, T_past+T_prst, dtype=torch.long, device=prst_idxs.device)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        embs = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            if (T_prst == 1):
                skip_first = 0
            else:
                skip_first = 1
            predictions = logits[:, T_past+skip_first:, :].squeeze(1)
            targets = targets[:, skip_first:].view(-1)
            loss = F.cross_entropy(predictions, targets)
        return logits, loss, embs


    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer