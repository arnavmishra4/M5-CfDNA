"""
model.py — Pleiades hierarchical cfDNA transformer architecture.
Contains all model classes. No training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "Healthy", 1: "GBM", 2: "LGG", 3: "DMG_H3K27M"}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class DNATokenEmbedding(nn.Module):
    """Level 0 — Fragment level: token + position + genomic alignment embeddings."""

    CHR_MAP = {**{str(i): i for i in range(1, 23)}, "X": 23, "Y": 24, "M": 25, "MT": 25}

    VOCAB = {
        'A': 0, 'T': 1, 'C': 2, 'G': 3,
        '<m>': 4, '<cls>': 5, '<pad>': 6,
        '<dna>': 7, '<mdna>': 8, '<cfdna>': 9,
        '<um>': 10, '</cfdna>': 11,
    }

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.token_embedding    = nn.Embedding(len(self.VOCAB), d_model, padding_idx=6)
        self.chr_embedding      = nn.Embedding(26, d_model)
        self.mil_embedding      = nn.Embedding(250, d_model)
        self.kilo_embedding     = nn.Embedding(1000, d_model)
        self.ones_embedding     = nn.Embedding(1000, d_model)
        self.position_embedding = nn.Embedding(1024, d_model)
        self.norm               = nn.LayerNorm(d_model)

    def _encode_chr(self, chromosome) -> int:
        if isinstance(chromosome, int):
            return chromosome
        return self.CHR_MAP.get(str(chromosome), 0)

    def forward(self, tokens: list, chromosome, genomic_start_pos: int) -> torch.Tensor:
        device  = self.token_embedding.weight.device
        seq_len = len(tokens)

        token_ids = torch.tensor([self.VOCAB[t] for t in tokens], dtype=torch.long, device=device)
        positions = torch.arange(seq_len, dtype=torch.long, device=device)

        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        abs_pos  = genomic_start_pos + positions
        chr_idx  = torch.full((seq_len,), self._encode_chr(chromosome), dtype=torch.long, device=device)
        mil_idx  = (abs_pos // 1_000_000) % 250
        kilo_idx = (abs_pos // 1_000) % 1000
        ones_idx = abs_pos % 1000

        align_emb = (
            self.chr_embedding(chr_idx)
            + self.mil_embedding(mil_idx)
            + self.kilo_embedding(kilo_idx)
            + self.ones_embedding(ones_idx)
        )

        return self.norm(tok_emb + pos_emb + align_emb)


class RegionEmbedding(nn.Module):
    """Level 1 — Region level: adds positional info to fragment CLS vectors."""

    def __init__(self, d_model: int = 512, max_fragments: int = 2048):
        super().__init__()
        self.fragment_pos_embedding = nn.Embedding(max_fragments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fragment_cls_vectors: torch.Tensor) -> torch.Tensor:
        device      = self.fragment_pos_embedding.weight.device
        n_fragments = fragment_cls_vectors.shape[0]
        positions   = torch.arange(n_fragments, dtype=torch.long, device=device)
        return self.norm(fragment_cls_vectors + self.fragment_pos_embedding(positions))


class SampleEmbedding(nn.Module):
    """Level 2 — Sample level: adds positional info to region CLS vectors."""

    def __init__(self, d_model: int = 512, max_regions: int = 512):
        super().__init__()
        self.region_pos_embedding = nn.Embedding(max_regions, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, region_cls_vectors: torch.Tensor) -> torch.Tensor:
        device    = self.region_pos_embedding.weight.device
        n_regions = region_cls_vectors.shape[0]
        positions = torch.arange(n_regions, dtype=torch.long, device=device)
        return self.norm(region_cls_vectors + self.region_pos_embedding(positions))


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.n_head       = n_head
        self.attn_dim     = d_model // n_head
        self.causal       = causal
        self.attn_dropout = nn.Dropout(dropout)

        self.W_Q = nn.Parameter(torch.empty(n_head, d_model, self.attn_dim))
        self.W_K = nn.Parameter(torch.empty(n_head, d_model, self.attn_dim))
        self.W_V = nn.Parameter(torch.empty(n_head, d_model, self.attn_dim))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))

        nn.init.xavier_normal_(self.W_Q)
        nn.init.xavier_normal_(self.W_K)
        nn.init.xavier_normal_(self.W_V)
        nn.init.xavier_normal_(self.W_O)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_expanded = x.unsqueeze(1)  # [B, 1, S, d_model]

        Q = torch.matmul(x_expanded, self.W_Q)  # [B, H, S, attn_dim]
        K = torch.matmul(x_expanded, self.W_K)
        V = torch.matmul(x_expanded, self.W_V)

        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=self.causal,
        )  # [B, H, S, attn_dim]

        output = output.transpose(1, 2).contiguous()        # [B, S, H, attn_dim]
        output = output.view(batch_size, seq_len, d_model)  # [B, S, d_model]
        return torch.matmul(output, self.W_O)


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.norm(x + sublayer_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, d_ff: int = 2048,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.mha      = MultiHeadAttention(d_model, n_head, dropout, causal)
        self.ff       = FeedForward(d_model, d_ff, dropout)
        self.addnorm1 = AddAndNorm(d_model)
        self.addnorm2 = AddAndNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addnorm1(x, self.dropout(self.mha(x)))
        x = self.addnorm2(x, self.dropout(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, d_ff: int = 2048,
                 n_layers: int = 6, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_head, d_ff, dropout, causal)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Pleiades — three-level hierarchical model
# ---------------------------------------------------------------------------

class PleiadesBase(nn.Module):
    """Level 0: fragment-level encoder. Returns CLS vector per fragment."""

    def __init__(self, d_model: int = 512, n_head: int = 8, d_ff: int = 2048,
                 n_layers: int = 6, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.embedding = DNATokenEmbedding(d_model)
        self.encoder   = Encoder(d_model, n_head, d_ff, n_layers, dropout, causal)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, tokens: list, chromosome, genomic_start_pos: int) -> torch.Tensor:
        x   = self.embedding(tokens, chromosome, genomic_start_pos).unsqueeze(0)
        cls = self.cls_token.expand(1, 1, -1)
        x   = checkpoint(self.encoder, torch.cat([cls, x], dim=1), use_reentrant=False)
        return x[:, 0, :]  # [1, d_model]


class PleiadesRegion(nn.Module):
    """Level 1: region-level encoder. Aggregates fragment CLS vectors."""

    def __init__(self, d_model: int = 512, n_head: int = 8, d_ff: int = 2048,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = RegionEmbedding(d_model)
        self.encoder   = Encoder(d_model, n_head, d_ff, n_layers, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, fragment_cls_vectors: torch.Tensor) -> torch.Tensor:
        x   = self.embedding(fragment_cls_vectors).unsqueeze(0)
        cls = self.cls_token.expand(1, 1, -1)
        x   = self.encoder(torch.cat([cls, x], dim=1))
        return x[:, 0, :]  # [1, d_model]


class PleiadesSample(nn.Module):
    """Level 2: sample-level encoder + classifier head."""

    def __init__(self, d_model: int = 512, n_head: int = 8, d_ff: int = 2048,
                 n_layers: int = 2, n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding  = SampleEmbedding(d_model)
        self.encoder    = Encoder(d_model, n_head, d_ff, n_layers, dropout)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, region_cls_vectors: torch.Tensor) -> torch.Tensor:
        x   = self.embedding(region_cls_vectors).unsqueeze(0)
        cls = self.cls_token.expand(1, 1, -1)
        x   = self.encoder(torch.cat([cls, x], dim=1))
        return self.classifier(x[:, 0, :])  # [1, n_classes]


class Pleiades(nn.Module):
    """
    Full hierarchical model: fragment → region → sample → class logits.

    Input format (patient_data dict):
        {
            "regions": {
                "region_key": {
                    "chromosome": "1",
                    "genomic_start": 123456,
                    "fragments": [
                        {"tokens": ["<cfdna>", "A", "T", ...], ...},
                        ...
                    ]
                },
                ...
            }
        }
    """

    MAX_FRAGS_PER_REGION = 64

    def __init__(self, d_model: int = 512, n_head: int = 8, d_ff: int = 2048,
                 base_layers: int = 6, region_layers: int = 4, sample_layers: int = 2,
                 n_classes: int = 4, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.base   = PleiadesBase(d_model, n_head, d_ff, base_layers, dropout, causal)
        self.region = PleiadesRegion(d_model, n_head, d_ff, region_layers, dropout)
        self.sample = PleiadesSample(d_model, n_head, d_ff, sample_layers, n_classes, dropout)

    def forward(self, patient_data: dict, use_cached: bool = False) -> torch.Tensor:
        device = next(self.parameters()).device
        region_cls_list = []

        for region_data in patient_data['regions'].values():
            fragments = region_data.get('fragments', [])[:self.MAX_FRAGS_PER_REGION]
            if not fragments:
                continue

            chromosome    = region_data['chromosome']
            genomic_start = region_data['genomic_start']

            fragment_cls_list = []
            for frag in fragments:
                if use_cached:
                    cls_vec = torch.tensor(frag['cls_vector'], device=device)
                else:
                    cls_vec = self.base(frag['tokens'], chromosome, genomic_start)
                fragment_cls_list.append(cls_vec.squeeze(0))

            if not fragment_cls_list:
                continue

            region_cls = self.region(torch.stack(fragment_cls_list))
            region_cls_list.append(region_cls.squeeze(0))

        if not region_cls_list:
            raise ValueError("Patient has no valid regions — check JSON structure.")

        return self.sample(torch.stack(region_cls_list))