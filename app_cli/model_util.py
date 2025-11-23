# inference_app.py
import torch
from transformers import AutoTokenizer
import torch.nn as nn

class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        d_model: int = 256,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.pad_id = pad_id

# 모델 세팅

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = nn.Dropout(config.dropout)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

        mask = torch.tril(torch.ones(config.max_len, config.max_len))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_len, config.max_len)
        )
    def forward(self, x, attn_mask=None):
        # x: (B, T, C)
        B, T, C = x.size()
        H = self.n_head
        head_dim = C // H

        qkv = self.qkv(x)              # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)  # (B, T, C) each

        q = q.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        k = k.view(B, T, H, head_dim).transpose(1, 2)
        v = v.view(B, T, H, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, T, T)

        causal_mask = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))

        if attn_mask is not None:
            # attn_mask: (B, T) → (B, 1, 1, T)
            pad_mask = attn_mask.view(B, 1, 1, T)
            att = att.masked_fill(pad_mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v                    # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # (선택) 입력 임베딩과 출력 head weight tying
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, attn_mask=None):
        # idx: (B, T)
        B, T = idx.size()
        device = idx.device

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand(B, T)  # (B, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

device = "cpu"  # 강제 CPU

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pad_id = tokenizer.pad_token_id
vocab_size = tokenizer.vocab_size
max_len = 512

config = GPTConfig(
    vocab_size=vocab_size,
    n_layer=4,
    n_head=4,
    d_model=256,
    d_ff=1024,
    max_len=max_len,
    dropout=0.1,
    pad_id=pad_id,
)


model = MiniGPT(config)
state = torch.load("lotto_gpt_best.pt", map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256, device="cpu"):
    model.eval()
    max_len = model.config.max_len
    eos_id = tokenizer.eos_token_id

    enc = tokenizer(prompt, return_tensors="pt")
    x = enc["input_ids"].to(device)

    for _ in range(max_new_tokens):
        if x.size(1) > max_len:
            x = x[:, -max_len:]

        logits = model(x)
        last_logits = logits[:, -1, :]

        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_token = next_id.item()

        if next_token == eos_id:
            break

        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text

if __name__ == "__main__":
    example_prompt = (
        "money=5000\n"
        "winning=1,2,3,4,5,6\n"
        "bonus=7\n"
        "###\n"
    )
    out = generate_text(model, tokenizer, example_prompt, device=device)
    print("=== SAMPLE ===")
    print(out)