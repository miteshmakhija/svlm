"""Offline top-k knowledge-distillation loss.

The teacher's distribution is stored as its top-k token ids and log-probabilities at each target
position (about 120 bytes per token for k=20). The student's distribution is restricted to the
same k tokens and both are renormalised, then compared with forward KL(teacher || student).

    loss = alpha * CE(student, target tokens) + (1 - alpha) * T^2 * KL_topk
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_kl(
    student_logits: torch.Tensor,  # [N, V]
    topk_ids: torch.Tensor,        # [N, k] long
    topk_logprobs: torch.Tensor,   # [N, k] teacher log-probs at T=1
    temperature: float = 1.0,
) -> torch.Tensor:
    T = float(temperature)
    vocab = student_logits.size(-1)
    valid = topk_ids < vocab                      # teacher vocab can be padded larger
    ids = topk_ids.clamp(max=vocab - 1)
    s_logp = F.log_softmax(student_logits.float() / T, dim=-1).gather(-1, ids)
    neg_inf = torch.finfo(s_logp.dtype).min
    s_logp = s_logp.masked_fill(~valid, neg_inf)
    s_logp = s_logp - torch.logsumexp(s_logp, dim=-1, keepdim=True)

    t_logits = (topk_logprobs.float() / T).masked_fill(~valid, neg_inf)
    t_logp = t_logits - torch.logsumexp(t_logits, dim=-1, keepdim=True)
    t_p = t_logp.exp()
    kl = (t_p * (t_logp - s_logp)).masked_fill(~valid, 0.0).sum(-1)
    return kl.mean() * (T * T)


def kd_loss(
    student_logits: torch.Tensor,  # [N, V] at target positions
    target_ids: torch.Tensor,      # [N]
    topk_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    ce = F.cross_entropy(student_logits.float(), target_ids)
    kl = topk_kl(student_logits, topk_ids, topk_logprobs, temperature)
    loss = alpha * ce + (1.0 - alpha) * kl
    return loss, {"ce": ce.detach().item(), "kl": kl.detach().item()}
