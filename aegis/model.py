"""The trajectory model (Methods: Model).

The architecture is Delphi's (Shmatko et al., Nature 2025; MIT licence, github.com/gerstung-lab/Delphi): GPT-2
blocks with 12 layers, 12 heads and 120-dimensional embeddings; the age is encoded with sine and cosine functions
and then a learned linear map; attention is masked so that no token sees later tokens or tokens recorded at the
same time as the target it predicts; a competing-exponentials head gives one rate per token, which tells both which
event comes next (softmax over the rates) and when (an exponential waiting time with the total rate).

AEGIS changes four things (Methods), the first three in this file:
  1. per-person loss: the token losses are averaged within each person (`loss_unit='person'`); Delphi averages
     over all tokens of a batch, so one person's gradient depends on the other people of the batch, which breaks
     the per-person clipping of DP-SGD;
  2. ICD-10 hierarchy: each code embedding is the sum of a code, a block and a chapter vector (`hier_emb`), in the
     input and (tied) in the output layer, so that rare codes borrow strength from their neighbours;
  3. public starting rate: the output bias starts so that the rates of all predicted tokens sum to one event per
     year (`init_rate='public'`); Delphi starts every token at about one event per day;
  4. the fixed vocabulary (data.Vocab).
For the ablation, loss_unit='token' and init_rate='delphi' switch changes 1 and 3 back to Delphi's behaviour.
"""
import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import INPUT_ONLY, PAD

# logit of a token that cannot happen: large and finite, as -inf turns into nan in the softmax gradients
NEG = -1e9


@dataclass
class Config:
    """Settings of an AEGIS model; saved with every checkpoint, in this field order."""
    vocab_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    # days: smoothing constant in the waiting-time term (Delphi's value)
    t_min: float = 0.1
    # public starting rate (change 3)
    init_events_per_year: float = 1.0
    # ICD-10 hierarchy embeddings (change 2)
    hier_emb: bool = False
    block_ids: list = None
    chapter_ids: list = None
    # rate calibration: log of one factor on every rate (synthetic.calibrate)
    log_scale: float = 0.0
    # 'person' (AEGIS) or 'token' (Delphi's weighting; ablation only)
    loss_unit: str = 'person'
    # with the 'token' unit a person's loss is the SUM of their token losses / 20, a fixed public constant, because
    # a normaliser derived from the data (the mean history length) would leak
    tok_norm: float = 20.0
    # 'public' (AEGIS) or 'delphi' (output bias 0: about one event per day per token)
    init_rate: str = 'public'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Config from a saved dict, also from checkpoints of the analysis server.

        Those checkpoints carry options that were off in every run of the paper (a prior, a time weight, tie
        masking); we accept them at their off values and refuse a checkpoint that turned any of them on, because
        this code does not implement them and would silently give a different model.
        """
        names = {f.name for f in fields(cls)}
        # options of the server code and the value at which they do nothing
        unused = {'prior': False, 'time_weight': 1.0, 'mask_ties': True}
        # object() equals nothing, so an option unknown to both lists is refused too
        bad = [(k, v) for k, v in d.items() if k not in names and unused.get(k, object()) != v]
        if bad:
            raise ValueError('checkpoint uses %s=%r, which this code does not implement' % bad[0])
        return cls(**{k: v for k, v in d.items() if k in names})


# model sizes: Delphi-2M (the paper's models) and a tiny one for the self-test on dummy data
PRESETS = {'delphi2m': dict(n_layer=12, n_head=12, n_embd=120),
           'tiny': dict(n_layer=2, n_head=2, n_embd=32)}


def _public_start(bias, c, n_pred):
    """Start every predicted token at an equal share of one event per year (change 3)."""
    nn.init.constant_(bias, math.log(c.init_events_per_year / 365.25 / n_pred))


def _delphi_start(bias, c, n_pred):
    """Keep Delphi's bias of 0, about one event per day per token (ablation)."""
    return None


# init_rate -> how the output bias starts
START_RATES = {'public': _public_start, 'delphi': _delphi_start}


def _per_person(n, c):
    """Divide a person's summed token losses by their number of targets (equation (2)); 0 targets count as 1."""
    return n.clamp(min=1)


def _per_token(n, c):
    """Divide by a public constant instead, Delphi's per-token weighting (ablation)."""
    return c.tok_norm


# loss_unit -> what the summed token losses of a person are divided by
LOSS_DIVISOR = {'person': _per_person, 'token': _per_token}


class AgeEncoding(nn.Module):
    """Sine and cosine encoding of the age in years, then a learned linear map, as in Delphi.

    The frequencies are those of the transformer position encoding; using years rather than days keeps the
    slowest wave longer than a lifetime.
    """

    def __init__(self, d):
        super().__init__()
        self.register_buffer('div', torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d)))
        self.lin = nn.Linear(2 * len(self.div), d, bias=False)

    def forward(self, age):
        angle = (age / 365.25).unsqueeze(-1) * self.div
        return self.lin(torch.cat([torch.sin(angle), torch.cos(angle)], -1))


class Block(nn.Module):
    """One GPT-2 block: masked multi-head self-attention and a GELU MLP, each with a pre-layer norm and a residual."""

    def __init__(self, c):
        super().__init__()
        d, b = c.n_embd, c.bias
        self.h, self.d = c.n_head, d
        # the layers are created in this order: it fixes the draws of their initial weights
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=b)
        self.proj = nn.Linear(d, d, bias=b)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=b)
        self.fc2 = nn.Linear(4 * d, d, bias=b)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask):
        B, T, _ = x.shape

        def heads(t):
            return t.view(B, T, self.h, self.d // self.h).transpose(1, 2)

        q, k, v = map(heads, self.qkv(self.ln1(x)).split(self.d, -1))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d // self.h)
        att = att.masked_fill(~mask, NEG).softmax(-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, self.d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)), approximate='tanh')))


class AegisModel(nn.Module):
    """Delphi's transformer with the AEGIS changes: per-person loss, ICD-10 hierarchy and public starting rate.

    The output of every position is one logit per token, the log of its rate per day. Tokens the model never
    predicts (padding, sex, lifestyle) get the logit NEG. Training goes through forward() with targets, which returns
    the loss per person, so that Opacus can compute and clip each person's gradient.

    Parameters
    ----------
    c : Config
    """

    def __init__(self, c):
        super().__init__()
        self.c, d = c, c.n_embd
        # creation order matters: each layer draws its initial weights from the global torch generator
        self.wte = nn.Embedding(c.vocab_size, d)
        self.wae = AgeEncoding(d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, c.vocab_size, bias=True)
        if c.tie_weights:
            self.wte.weight = self.head.weight
        if c.hier_emb:
            self._hierarchy(c, d, max(c.block_ids) + 1, max(c.chapter_ids) + 1)
        self.apply(self._init)
        START_RATES[c.init_rate](self.head.bias, c, c.vocab_size - len(INPUT_ONLY))
        # GPT-2's scaled start for the layers that write into the residual stream, so that its variance does not
        # grow with depth
        for name, p in self.named_parameters():
            if name.endswith(('proj.weight', 'fc2.weight')):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))
        not_predicted = torch.zeros(c.vocab_size, dtype=torch.bool).index_fill_(0, torch.tensor(INPUT_ONLY), True)
        self.register_buffer('not_predicted', not_predicted)

    def _hierarchy(self, c, d, nb, nc):
        """Block and chapter vectors, in the input and (tied) in the output layer."""
        self.register_buffer('block_id', torch.tensor(c.block_ids, dtype=torch.long))
        self.register_buffer('chapter_id', torch.tensor(c.chapter_ids, dtype=torch.long))
        self.wbe = nn.Embedding(nb, d)
        self.wce = nn.Embedding(nc, d)
        self.head_b = nn.Linear(d, nb, bias=False)
        self.head_c = nn.Linear(d, nc, bias=False)
        self.wbe.weight, self.wce.weight = self.head_b.weight, self.head_c.weight

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def masks(self, x, a, b=None):
        """Attention mask and the age from which each waiting time is measured.

        A query may attend to earlier keys that are not padding. With the target ages b, it may also not attend to
        a key recorded at the age of its own target: records on the same day (the lifestyle tokens of one visit,
        several codes of one admission) would otherwise tell the model what comes next. A query left with no key
        attends to itself, so that the softmax is defined.

        Parameters
        ----------
        x, a : torch.Tensor [B, T]
            Input tokens and ages (days).
        b : torch.Tensor [B, T], optional
            Target ages.

        Returns
        -------
        mask : torch.Tensor of bool [B, 1, T, T]
        last : torch.Tensor [B, T] or None
            Per query, the age of the last visible key (None without b).
        """
        B, T = x.shape
        dev = x.device
        allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None] & (x != PAD)[:, None, :]
        if b is not None:
            allowed = allowed & (a[:, None, :] != b[:, :, None])
        eye = torch.eye(T, dtype=torch.bool, device=dev)[None]
        empty = ~allowed.any(-1, keepdim=True)
        last = None
        if b is not None:
            pos = torch.arange(T, device=dev).view(1, 1, T).expand(B, T, T)
            last_index = torch.where(allowed, pos, torch.full_like(pos, -1)).max(-1).values
            # a query that sees no key measures its waiting time from its own age
            own = torch.arange(T, device=dev).expand(B, T)
            last = torch.gather(a, 1, torch.where(last_index < 0, own, last_index))
        return (allowed | (empty & eye))[:, None], last

    def forward(self, x, a, b=None, y=None):
        """Logits [B, T, V] (log-rates per day), or with targets y the per-person loss [B].

        Training goes through forward() so that Opacus' per-person gradient hooks see every parameter.
        """
        if y is None:
            return self.logits(x, a, b)
        return self.person_loss(x, a, y, b)[0]

    def logits(self, x, a, b=None, mask=None):
        """Log-rate per day of every token at every position, [B, T, V]; see masks() for the attention."""
        if mask is None:
            mask = self.masks(x, a, b)[0]
        if self.c.hier_emb:
            e = self.wte(x) + self.wbe(self.block_id[x]) + self.wce(self.chapter_id[x])
        else:
            e = self.wte(x)
        h = self.drop(e + self.wae(a))
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_f(h)
        if self.c.hier_emb:
            logits = self.head(h) + self.head_b(h)[..., self.block_id] + self.head_c(h)[..., self.chapter_id]
        else:
            logits = self.head(h)
        # one constant on every logit multiplies every rate by exp(log_scale): the softmax over the next event and
        # every ranking (so every AUC) are unchanged, only the overall rate level moves
        if self.c.log_scale:
            logits = logits + self.c.log_scale
        return logits.masked_fill(self.not_predicted, NEG)

    def token_losses(self, x, a, y, b, parts=False):
        """Delphi's loss per token: cross-entropy of the next token plus the exponential waiting-time term.

        With the total rate lambda (the sum of the token rates) and the waiting time dt, the time term is
        -(log lambda - lambda (dt + t_min)), the negative log-likelihood of an exponential waiting time. We smooth
        lambda as 1 / (1 / lambda + t_min), Delphi's form, which caps the total rate at 1 / t_min (10 events a
        day) and keeps the term finite, and we count every waiting time as at least one day.

        Parameters
        ----------
        x, a, y, b : torch.Tensor [B, T]
            Input tokens and ages, target tokens and ages.
        parts : bool, default False
            Also return the two terms.

        Returns
        -------
        loss, valid : torch.Tensor [B, T]
            Loss per token (0 where there is no target) and the mask of valid targets; with parts=True also the
            cross-entropy and the time term.
        """
        mask, last = self.masks(x, a, b)
        valid = (y != PAD) & ~self.not_predicted[y]
        logits = self.logits(x, a, b, mask)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='none').view(y.shape)
        log_total = torch.logsumexp(logits, -1)
        # smoothed log total rate
        lse = -torch.logaddexp(-log_total, torch.full_like(log_total, math.log(self.c.t_min)))
        # waiting time of at least one day
        tl = -(lse - torch.exp(lse + torch.log(torch.clamp(b - last, min=1.0) + self.c.t_min)))
        if parts:
            return (ce + tl) * valid, valid, ce * valid, tl * valid
        return (ce + tl) * valid, valid

    def person_loss(self, x, a, y, b):
        """Loss per person (equation (2)) and the number of targets.

        A person's token losses are summed and divided by their number of targets, so that each person weighs the
        same in a step whatever the length of their history, and a person's loss depends on no one else in the
        batch, which per-person clipping in DP-SGD needs. A person without targets gets 0.

        Returns
        -------
        loss : torch.Tensor [B]
        n : torch.Tensor of int [B]
            Number of valid targets per person.
        """
        token_loss, valid = self.token_losses(x, a, y, b)
        n = valid.sum(1)
        return token_loss.sum(1) / LOSS_DIVISOR[self.c.loss_unit](n, self.c), n

    def loss(self, x, a, y, b):
        return self.person_loss(x, a, y, b)[0].mean()


def with_max_events(model, extra):
    """The model and the longest history (in tokens) it was trained with."""
    return model, extra['train_cfg']['max_events']


def save_model(model, path, extra=None):
    torch.save({'config': model.c.to_dict(), 'state': model.state_dict(), 'extra': extra or {}}, path)


def load_model(path, device='cpu'):
    """Load a checkpoint written by save_model; returns the model in eval mode and the saved extras."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AegisModel(Config.from_dict(checkpoint['config']))
    model.load_state_dict(checkpoint['state'])
    return model.to(device).eval(), checkpoint.get('extra', {})
