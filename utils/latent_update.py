import torch

from edit.erase import (
    erase_inpaint_then_inv_latent,
    erase_loss_global_optim_latent,
    erase_loss_optim_latent,
    erase_masked_renoise_from_x0,
    erase_masked_random_noise_from_x0,
)
from edit.compose import (
    compose_gaussian_mix,
    compose_norm_mix,
    compose_norm_mix_overlap_avg,
    compose_norm_feather_mix,
    compose_weighted_transition_mix,
    compose_weighted_mix,
)


def update_latent(latents, loss, step_size):
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
    return latents - step_size * grad_cond


_ERASE_FNS = {
    "loss_optim": erase_loss_optim_latent,
    "loss_global": erase_loss_global_optim_latent,
    "masked_renoise": erase_masked_renoise_from_x0,
    "masked_random_noise": erase_masked_random_noise_from_x0,
    "inpaint_then_inv": erase_inpaint_then_inv_latent,
}

_COMPOSE_FNS = {
    "weighted": compose_weighted_mix,
    "weighted_transition": compose_weighted_transition_mix,
    "norm": compose_norm_mix,
    "norm_overlap_avg": compose_norm_mix_overlap_avg,
    "norm_feather": compose_norm_feather_mix,
    "gaussian": compose_gaussian_mix,
}


def _update_latent(x, erase=None, compose=None, *, erase_kwargs=None, compose_kwargs=None, strict=True):
    """
    x: latent 입력
    erase: str 또는 None
    compose: str 또는 None
    erase_kwargs/compose_kwargs: 선택된 함수에 전달할 kwargs
    """
    out = x

    if erase:
        if erase not in _ERASE_FNS:
            if strict:
                raise ValueError(f"Unknown erase method: {erase}")
            return out
        kwargs = erase_kwargs or {}
        try:
            out = _ERASE_FNS[erase](out, **kwargs)
        except TypeError:
            if strict:
                raise
        
    if compose:
        if compose not in _COMPOSE_FNS:
            if strict:
                raise ValueError(f"Unknown compose method: {compose}")
            return out
        kwargs = compose_kwargs or {}
        try:
            out = _COMPOSE_FNS[compose](out, **kwargs)
        except TypeError:
            if strict:
                raise
        
    return out
