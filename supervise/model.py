# Model part
import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 235)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_logits = self._tower(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask

import math
from torch.nn import functional as F
# import torch.distributed as dist

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, maxlen):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)  # token embedding
        # pe = torch.zeros(maxlen, d_model).float()
        # pe.require_grad = False
        # position = torch.arange(0, maxlen).float().unsqueeze(1)
        # div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        embedding = self.tok_embed(x)#  + self.pe[:, :x.size(1)]
        return self.norm(embedding)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen):
        super().__init__()
        assert d_model % nhead == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(drop)
        self.resid_dropout = nn.Dropout(drop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen))
        # rpe = torch.zeros(1, nhead, maxlen, maxlen)
        # for i in range(1, maxlen):
        #     rpe = rpe - torch.tril(torch.ones(maxlen, maxlen), diagonal=-i).view(1, 1, maxlen, maxlen)
        # for i in range(nhead):
        #     rpe[0, i] = rpe[0, i] * 2 **(-8 / nhead * (i + 1))
        # self.register_buffer("RPE", rpe)
        self.n_head = nhead
        self.n_embd = d_model

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))# + self.RPE[:, :, :T, :T]
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # if mask is not None:
        #     att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, d_model, nhead, drop, maxlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, nhead=nhead, drop=drop, maxlen=maxlen)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(d_model, 4 * d_model),
            c_proj  = nn.Linear(4 * d_model, d_model),
            act     = NewGELU(),
            dropout = nn.Dropout(drop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class Encoder(nn.Module):
    """ GPT Language Model """
    def __init__(self, d_model=256, vocab_size=44, drop=0.1, maxlen=20, nhead=4, num_layer=3):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            embedding = Embedding(d_model=d_model, vocab_size=vocab_size, maxlen=maxlen),
            drop = nn.Dropout(drop),
            h = nn.ModuleList([Block(d_model=d_model, nhead=nhead, drop=drop, maxlen=maxlen) for _ in range(num_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, 235, bias=True)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))
        # if dist.get_rank() == 0:
        #     print("number of parameters: %.2fM" % (n_params/1e6,))

    def obs2logit(self, idx):
        b, t = idx.size()
        mask = (idx > 0).unsqueeze(1).repeat(1, idx.size(1), 1).unsqueeze(1)
        # forward the GPT model itself
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x, mask)
        x = self.transformer.ln_f(x[:, 0])
        logits = self.lm_head(x)
        return logits

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].long()
        action_logits = self.obs2logit(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
