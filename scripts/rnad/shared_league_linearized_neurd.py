"""Research-only autograd-light NeuRD surrogate, not a different update rule.

For every legal-action vector, the clipped force is stop-gradient in the
reference NeuRD surrogate. Therefore its exact local gradient is the
mean-centered constant coefficient -(force-mean(force))/candidate_count.
Batch all (independent) vectors into ONE differentiable dot product instead
of building hundreds of repeated scalar autograd graphs.

The original scalar loss value is restored with detached constants. All legal
actions, clipped advantages, V-trace correction and reference beta gate remain.
"""
from __future__ import annotations

import torch


def linearized_neurd_loss_batch(entries, *, beta=2.0, q_clip=10.0,
                                logit_step=3e-4):
    """Mean of the existing stop-grad NeuRD surrogates over same-phase rows.

    entries: (live logits, detached all-action Q, clipped V-trace c).
    A caller keeps the original 1/3 phase balancing and 1/32 Match gradient.
    """
    if not entries:
        raise ValueError("No legal NeuRD entries")
    if not 0 < logit_step <= 1 or beta < 0 or q_clip <= 0:
        raise ValueError("Invalid clipped NeuRD parameters")
    live, coefficients, constants = [], [], []
    with torch.no_grad():
        for logits, q_targets, is_c in entries:
            if (not isinstance(is_c, (int, float)) or
                not 0 <= is_c <= 1):
                raise ValueError("Invalid clipped NeuRD V-trace correction")
            if (logits.ndim != 1 or logits.shape != q_targets.shape or
                logits.numel() < 1 or
                not torch.isfinite(logits).all() or
                not torch.isfinite(q_targets).all()):
                raise ValueError("Invalid all-legal-action NeuRD vectors")
            q = q_targets.detach().to(logits).clamp(-q_clip, q_clip)
            frozen = logits.detach()
            probs = torch.softmax(frozen, dim=-1)
            advantage = (q - torch.sum(probs*q)) * is_c
            centered = frozen - frozen.mean()
            if beta > 0:
                proposed = centered + logit_step * advantage
                force = torch.where(proposed.abs() <= beta, advantage,
                                    torch.zeros_like(advantage))
            else:
                force = advantage
            # d/dlogits [ -(logits-mean(logits))*force / n ]
            # = -(force-mean(force)) / n, since force is detached.
            coeff = -(force - force.mean()) / logits.numel()
            reference = -(centered*force).sum()/logits.numel()
            offset = reference - (frozen*coeff).sum()
            live.append(logits)
            coefficients.append(coeff)
            constants.append(offset)
    # ONE differentiable operation across all variable-length legal supports.
    # Detached offsets retain the original scalar objective for diagnostics.
    return ((torch.cat(live) * torch.cat(coefficients)).sum() +
            torch.stack(constants).sum()) / len(entries)
