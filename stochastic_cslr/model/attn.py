import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, rpe_q=None, rpe_v=None):
        """
        Args:
            q: query (*, query_len, dim)
            k: key (*, key_len, dim)
            v: value (*, key_len, dim)
            mask: (*, query_len, key_len), True will be masked out
            rpe_q : (query_len, key_len, dim)
            rpe_v : (query_len, key_len, dim)
        Returns:
            context: (*, query_len, dim)
            alignment: (*, query_len, key_len)
        """
        dim = q.shape[-1]

        q /= dim ** 0.5
        energy = q @ k.transpose(-2, -1)

        if rpe_q is not None:
            energy += torch.einsum("...qd,qkd->...qk", q, rpe_q)

        if mask is not None:
            energy = energy.masked_fill(mask, np.NINF)

        alignment = torch.softmax(energy, dim=-1)
        context = self.dropout(alignment) @ v

        if rpe_v is not None:
            context += torch.einsum("...qk,qkd->...qd", alignment, rpe_v)

        return context, alignment


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout, rpe_k=0):
        assert (
            dim % heads == 0
        ), "dim should be a multiple of heads, \
            got {} and {}".format(
            dim, heads
        )

        super().__init__()

        self.dim = dim
        self.heads = heads

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.rpe_k = rpe_k
        if rpe_k > 0:
            self.rpe_w = nn.Embedding(rpe_k * 2 + 1, 2 * dim // heads)

        self.attn = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query (batch, query_len, dim)
            k: key (batch, key_len, dim)
            v: value (batch, key_len, dim)
            mask: (batch, query_len, key_len)
        Returns:
            context: (batch, query_len, dim)
            alignment: (bs, head, ql, kl)
        """

        bs, ql, kl = (*q.shape[:2], k.shape[1])

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        split_heads = lambda x: rearrange(x, "b t (h d) -> b h t d", h=self.heads)
        q, k, v = map(split_heads, (q, k, v))

        # add head dim for mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        if self.rpe_k > 0:
            distance = self.relative_distance(max(ql, kl), self.rpe_k)
            distance = distance[:ql, :kl].to(q.device)
            rpe_q, rpe_v = self.rpe_w(distance).chunk(2, dim=-1)
            context, alignment = self.attn(q, k, v, mask, rpe_q, rpe_v)
        else:
            context, alignment = self.attn(q, k, v, mask)

        # swap len and head back
        context = rearrange(context, "b h t d -> b t (h d)")
        context = self.fc(context)

        return context, alignment

    @staticmethod
    def relative_distance(length, k):
        indices = torch.arange(length)
        indices = indices.unsqueeze(1).expand(-1, length)
        distance = indices - indices.transpose(0, 1)
        distance = distance.clamp(-k, k) + k
        return distance
