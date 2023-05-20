# Botzone interaction
import numpy as np
import torch

from collections import defaultdict

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

import torch
from torch import nn
import math
from torch.nn import functional as F
# import torch.distributed as dist

class Head(nn.Module):
    def __init__(self, d_model=256):
        super(Head, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.linear(x)


class Embedding(nn.Module):
    def __init__(self, d_model, maxlen):
        super(Embedding, self).__init__()
        # self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)  # token embedding
        self.Type = nn.Embedding(11, d_model, padding_idx=0)
        self.Color = nn.Embedding(11, d_model, padding_idx=0)
        self.Num = nn.Embedding(10, d_model, padding_idx=0)
        self.Duplicate = nn.Embedding(10, d_model, padding_idx=0)
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
        embedding = self.Type(x[:, :, 0]) + self.Color(x[:, :, 1]) + self.Num(x[:, :, 2]) + self.Duplicate(x[:, :, 3])
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
    def __init__(self, d_model=256, vocab_size=214, drop=0.1, maxlen=20, nhead=4, num_layer=4):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            embedding = Embedding(d_model=d_model, maxlen=maxlen),
            drop = nn.Dropout(drop),
            h = nn.ModuleList([Block(d_model=d_model, nhead=nhead, drop=drop, maxlen=maxlen) for _ in range(num_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, 235, bias=True)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("number of parameters: %.2fM" % (n_params/1e6,))
        # if dist.get_rank() == 0:
        #     print("number of parameters: %.2fM" % (n_params/1e6,))

        # type   cls:1 seat:2 prevalent:3 hand:4 chi:5 peng:6 gang:7 shown:8 cur:9 begin:10
        # color  W:1 T:2 B:3 F1:4 F2:5 F3:6 F4:7 J1:8 J2:9 J3:10
        # num    1-9
        # duplicate: my_card: 1-4  shown: 5-9
        transfer = [[0, 0, 0, 0], [1,0,0,0]]
        for i in range(2, 4):
            transfer = transfer + [[i, j, 0, 0]for j in range(4, 8)]

        for i in range(1, 4):
            for j in range(1, 10):
                transfer = transfer + [[4, i, j, k]for k in range(1, 5)]
        for i in range(4, 11):
            transfer = transfer + [[4, i, 0, k]for k in range(1, 5)]

        for i in range(1, 4):
            for j in range(1, 10):
                transfer = transfer + [[5, i, j, 1]]
        for i in range(4, 11):
            transfer = transfer + [[5, i, 0, 1]]

        for i in range(1, 4):
            for j in range(1, 10):
                transfer = transfer + [[6, i, j, k] for k in range(1, 4)]
        for i in range(4, 11):
            transfer = transfer + [[6, i, 0, k] for k in range(1, 4)]

        for i in range(1, 4):
            for j in range(1, 10):
                transfer = transfer + [[7, i, j, k] for k in range(1, 5)]
        for i in range(4, 11):
            transfer = transfer + [[7, i, 0, k] for k in range(1, 5)]

        for i in range(1, 4):
            for j in range(1, 10):
                transfer = transfer + [[8, i, j, k] for k in range(5, 10)]
        for i in range(4, 11):
            transfer = transfer + [[8, i, 0, k] for k in range(5, 10)]

        for i in range(1, 4):
            for j in range(1, 10):
                transfer = transfer + [[9, i, j, 0]]
        for i in range(4, 11):
            transfer = transfer + [[9, i, 0, 0]]
        transfer.append([10, 0, 0, 0])
        
        self.register_buffer('dic', torch.tensor(transfer))

    def obs2logit(self, idx):
        b, t = idx.size()
        mask = (idx > 0).unsqueeze(1).repeat(1, idx.size(1), 1).unsqueeze(1)
        # forward the GPT model itself
        idx = self.dic[idx]
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x, mask)
        x = self.transformer.ln_f(x[:, 0])
        return self.lm_head(x)

    def forward(self, input_dict):
        obs = input_dict["observation"].long()
        action_logits = self.obs2logit(obs, Head)
        action_mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask

    # def forward(self, input_dict):
    #     obs = input_dict["observation"].float()
    #     hidden = self._tower(obs)
    #     logits = self._logits(hidden)
    #     mask = input_dict["action_mask"].float()
    #     inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
    #     masked_logits = logits + inf_mask
    #     value = self._value_branch(hidden)
    #     return masked_logits, value


class FeatureAgent():
    
    '''
    observation: 6*4*9
        (men+quan+hand4)*4*9
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    '''
    
    OBS_SIZE = 6
    ACT_SIZE = 235
    
    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}
    
    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
        self.curTile = -1
    
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        my_obs = [1, self.seatWind+2, self.prevalentWind+6] # 10, 3
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            for i in range(d[tile]):
                my_obs.append(self.OFFSET_TILE[tile]*4 + 10 +i) # 10 + 4 * 34 = 146
        for pack in self.packs[0]:
            if pack[0] == "CHI":
                x = self.OFFSET_TILE[pack[1]]
                for i in range(-1, 2):
                    my_obs.append(146+x+i) # 146 + 34 = 180
            if pack[0] == "PENG":
                x = self.OFFSET_TILE[pack[1]]
                for i in range(3):
                    my_obs.append(180+3*x+i) # 180 + 34*3 = 282
            if pack[0] == "GANG":
                x = self.OFFSET_TILE[pack[1]]
                for i in range(4):
                    my_obs.append(180+4*x+i) # 282 + 34*4 = 418, 3 + 4*4 + 2 = 21
        for i in self.TILE_LIST:
            my_obs.append(self.OFFSET_TILE[i]*5 + 418 + self.shownTiles[i]) # 418 + 5 * 34 = 588, 21 + 34 = 55
        if self.curTile != -1:
            my_obs.append(self.OFFSET_TILE[self.curTile] + 588) # 588 + 34 = 622, 56
        else:
            my_obs.append(622) # 623, 56
        while len(my_obs) < 56:
            my_obs.append(0)
        return {
            'observation': np.array(my_obs, dtype=np.int32),
            'action_mask': mask
        }
        # return {
        #     'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
        #     'action_mask': mask
        # }
    
    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : ] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = self.shownTiles[winTile] == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True

def obs2response(model, obs):
    logits = model({'is_training': False, 'obs': {'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)), 'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))}})
    action = logits.detach().numpy().flatten().argmax()
    response = agent.action2response(action)
    return response

import sys

if __name__ == '__main__':
    model = CNNModel()
    data_dir = '/data/rl.pkl'
    model.load_state_dict(torch.load(data_dir, map_location = torch.device('cpu')))
    input() # 1
    while True:
        request = input()
        while not request.strip(): request = input()
        request = request.split()
        if request[0] == '0':
            seatWind = int(request[1])
            agent = FeatureAgent(seatWind)
            agent.request2obs('Wind %s' % request[2])
            print('PASS')
        elif request[0] == '1':
            agent.request2obs(' '.join(['Deal', *request[5:]]))
            print('PASS')
        elif request[0] == '2':
            obs = agent.request2obs('Draw %s' % request[1])
            response = obs2response(model, obs)
            response = response.split()
            if response[0] == 'Hu':
                print('HU')
            elif response[0] == 'Play':
                print('PLAY %s' % response[1])
            elif response[0] == 'Gang':
                print('GANG %s' % response[1])
                angang = response[1]
            elif response[0] == 'BuGang':
                print('BUGANG %s' % response[1])
        elif request[0] == '3':
            p = int(request[1])
            if request[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')
            elif request[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif request[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, request[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')
            else:
                zimo = False
                if request[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, request[3]))
                elif request[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, request[-1]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    response = response.split()
                    if response[0] == 'Hu':
                        print('HU')
                    elif response[0] == 'Pass':
                        print('PASS')
                    elif response[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif response[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d '% seatWind + response)
                        response2 = obs2response(model, obs)
                        print(' '.join([response[0].upper(), *response[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + response)
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()