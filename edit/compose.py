import torch
import torch.nn.functional as F


# compose_norm_mix
# 마스크 기반 평균 합성(겹치면 평균)
def compose_norm_mix(x_t_bg, objs, eps=1e-6):
    """
    x_t_bg: [B,4,H,W]
    objs: obj.mask_tgt_lat, obj.xt_patch_full 필요
    """
    if not objs:
        return x_t_bg

    value = torch.zeros_like(x_t_bg)
    count = torch.zeros_like(x_t_bg[:, :1])

    for obj in objs:
        m = obj.mask_tgt_lat
        value = value + obj.xt_patch_full * m
        count = count + m

    m_union = (count > eps).float()
    avg = value / count.clamp_min(eps)
    x = x_t_bg * (1.0 - m_union) + avg * m_union
    return x


# compose_norm_mix_overlap_avg
# 비겹침은 hard overwrite, 겹침만 평균
def compose_norm_mix_overlap_avg(x_t_bg, objs, eps=1e-6):
    x = x_t_bg.clone()
    m_sum = torch.zeros_like(x_t_bg[:, :1])

    for obj in objs:
        m = obj.mask_tgt_lat
        x = x * (1.0 - m) + obj.xt_patch_full * m
        m_sum = m_sum + m

    overlap = (m_sum > 1.0).float()
    if overlap.sum() == 0:
        return x

    value = torch.zeros_like(x_t_bg)
    count = torch.zeros_like(x_t_bg[:, :1])
    for obj in objs:
        m = obj.mask_tgt_lat
        value = value + obj.xt_patch_full * m
        count = count + m

    avg = value / count.clamp_min(eps)
    x = x * (1.0 - overlap) + avg * overlap
    return x


# compose_gaussian_mix (TF-ICON 스타일: XOR)
def compose_gaussian_mix(x_t_bg, objs, t, scheduler):
    """
    x_t_bg: [B,4,H,W]
    objs: obj.mask_tgt_lat, obj.xt_patch_full 필요 (첫 두 개 사용)
    t: scheduler.timesteps에서 꺼낸 값
    scheduler: add_noise 사용
    """
    if len(objs) < 2:
        return compose_norm_mix(x_t_bg, objs)

    m_o = objs[0].mask_tgt_lat
    m_s = objs[1].mask_tgt_lat

    x = x_t_bg * (1.0 - torch.clamp(m_o + m_s, 0, 1))
    x = x + objs[0].xt_patch_full
    x = x + objs[1].xt_patch_full

    m_xor = (m_o > 0.5) ^ (m_s > 0.5)
    m_xor = m_xor.float()

    eps = torch.randn_like(x)
    noise_xt = scheduler.add_noise(torch.zeros_like(x), eps, t)
    x = x * (1.0 - m_xor) + noise_xt * m_xor
    return x


# --- New compose methods for cleaner pasting ---

def _dilate_mask(m, kernel_size: int, iters: int = 1):
    """Binary-ish dilation on [B,1,H,W] float masks using max_pool2d."""
    if kernel_size <= 1 or iters <= 0:
        return m
    pad = kernel_size // 2
    out = m
    for _ in range(iters):
        out = F.max_pool2d(out, kernel_size=kernel_size, stride=1, padding=pad)
    return out


def _blur_mask(m, kernel_size: int, iters: int = 1):
    """Simple box blur on [B,1,H,W] float masks using avg_pool2d."""
    if kernel_size <= 1 or iters <= 0:
        return m
    pad = kernel_size // 2
    out = m
    for _ in range(iters):
        out = F.avg_pool2d(out, kernel_size=kernel_size, stride=1, padding=pad)
    return out


def compose_norm_feather_mix(x_t_bg, objs, feather_kernel: int = 7, feather_iters: int = 1, beta: float = 8.0, eps: float = 1e-6):
    """
    Like weighted z-buffer mixing but with feathered (blurred) masks to reduce seams.
    """
    if not objs:
        return x_t_bg

    masks_soft = []
    for obj in objs:
        m = obj.mask_tgt_lat.clamp(0.0, 1.0)
        m = _blur_mask(m, feather_kernel, feather_iters).clamp(0.0, 1.0)
        masks_soft.append(m.clamp(eps, 1.0))

    m_stack = torch.stack(masks_soft, dim=0)  # [K,B,1,H,W]
    m_union = m_stack.max(dim=0).values
    m_bg = (1.0 - m_union).clamp(eps, 1.0)

    logits = [beta * torch.log(m_bg)]
    for obj, m in zip(objs, masks_soft):
        p = getattr(obj, "priority", 0.0)
        logits.append(beta * (p + torch.log(m.clamp(eps, 1.0))))

    stacked = torch.stack(logits, dim=0)  # [K+1,B,1,H,W]
    alphas = torch.softmax(stacked, dim=0)
    alpha_bg = alphas[0]
    alpha_list = [alphas[i + 1] for i in range(len(objs))]

    x = alpha_bg * x_t_bg
    for obj, a in zip(objs, alpha_list):
        x = x + a * obj.xt_patch_full
    return x


def compose_weighted_transition_mix(
    x_t_bg,
    objs,
    t,
    scheduler,
    transition_kernel: int = 7,
    transition_iters: int = 1,
    transition_strength: float = 1.0,
    beta: float = 8.0,
    eps: float = 1e-6,
):
    """
    Weighted z-buffer mix + transition-noise on the boundary/overlap region.
    This helps the diffusion steps re-synthesize seams instead of hard-pasting them.
    """
    if not objs:
        return x_t_bg

    # Z-buffer style mixing (same spirit as overlap_soft_zbuffer).
    masks = [o.mask_tgt_lat.clamp(eps, 1.0) for o in objs]
    m_stack = torch.stack(masks, dim=0)  # [K,B,1,H,W]
    m_union = m_stack.max(dim=0).values
    m_bg = (1.0 - m_union).clamp(eps, 1.0)

    logits = [beta * torch.log(m_bg)]
    for obj in objs:
        p = getattr(obj, "priority", 0.0)
        logits.append(beta * (p + torch.log(obj.mask_tgt_lat.clamp(eps, 1.0))))

    stacked = torch.stack(logits, dim=0)  # [K+1,B,1,H,W]
    alphas = torch.softmax(stacked, dim=0)
    alpha_bg = alphas[0]
    alpha_list = [alphas[i + 1] for i in range(len(objs))]

    x = alpha_bg * x_t_bg
    for obj, a in zip(objs, alpha_list):
        x = x + a * obj.xt_patch_full

    # Transition region: boundary of union + overlap areas.
    m_union_bin = (m_union > eps).float()
    m_dil = _dilate_mask(m_union_bin, transition_kernel, transition_iters)
    boundary = (m_dil > 0.5).float() * (1.0 - m_union_bin)

    overlap = (m_stack.sum(dim=0) > 1.0).float()
    m_transition = torch.clamp(boundary + overlap, 0.0, 1.0) * float(transition_strength)

    if m_transition.sum() == 0:
        return x

    eps_noise = torch.randn_like(x)
    noise_xt = scheduler.add_noise(torch.zeros_like(x), eps_noise, t)
    x = x * (1.0 - m_transition) + noise_xt * m_transition
    return x


# compose_weighted_mix
# alpha 기반 weighted 합성 + transition 노이즈
def compose_weighted_mix(x_t_bg, objs, alpha_list, alpha_bg, m_transition, t, scheduler):
    """
    x_t_bg: [B,4,H,W]
    objs: obj.xt_patch_full 필요
    alpha_list: objs와 동일 순서
    alpha_bg: [B,1,H,W]
    m_transition: [B,1,H,W]
    t: scheduler.timesteps에서 꺼낸 값
    """
    x = alpha_bg * x_t_bg
    for obj, a in zip(objs, alpha_list):
        x = x + a * obj.xt_patch_full

    eps = torch.randn_like(x)
    noise_xt = scheduler.add_noise(torch.zeros_like(x), eps, t)
    x = x * (1.0 - m_transition) + noise_xt * m_transition
    return x
