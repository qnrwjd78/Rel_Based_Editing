import torch


def _softmax_stack(logits):
    stacked = torch.stack(logits, dim=0)
    alphas = torch.softmax(stacked, dim=0)
    return [alphas[i] for i in range(alphas.shape[0])]


# overlap_soft_zbuffer (bg 포함)
# layer priority + log(mask) 를 softmax
# 결과: priority 높은 객체가 겹침에서 우선
# 반환: (alpha_bg, alpha_list)
def overlap_soft_zbuffer(objs, beta=8.0, p_bg=0.0, eps=1e-6):
    masks = [o.mask_tgt_lat.clamp(eps, 1.0) for o in objs]
    m_stack = torch.stack(masks, dim=0)  # [K,B,1,H,W]
    m_union = m_stack.max(dim=0).values
    m_bg = (1.0 - m_union).clamp(eps, 1.0)

    logits = [beta * (p_bg + torch.log(m_bg))]
    for obj in objs:
        m = obj.mask_tgt_lat.clamp(eps, 1.0)
        logits.append(beta * (obj.priority + torch.log(m)))

    stacked = torch.stack(logits, dim=0)  # [K+1,B,1,H,W]
    alphas = torch.softmax(stacked, dim=0)
    alpha_bg = alphas[0]
    alpha_list = [alphas[i + 1] for i in range(len(objs))]
    return alpha_bg, alpha_list


# overlap_attn_gate (bg 포함 권장)
# attention map을 softmax로 정규화
# 반환: (alpha_bg, alpha_list)
def overlap_attn_gate(attn_maps_by_obj, masks_by_obj=None, tau=0.2, eps=1e-6):
    if masks_by_obj is None:
        masks_by_obj = [torch.ones_like(a) for a in attn_maps_by_obj]

    masked_attn = [a * m for a, m in zip(attn_maps_by_obj, masks_by_obj)]
    logits = [a.clamp(eps) / max(tau, eps) for a in masked_attn]

    m_stack = torch.stack([m.clamp(eps, 1.0) for m in masks_by_obj], dim=0)
    m_union = m_stack.max(dim=0).values
    m_bg = (1.0 - m_union).clamp(eps, 1.0)
    logits = [torch.log(m_bg)] + logits

    stacked = torch.stack(logits, dim=0)
    alphas = torch.softmax(stacked, dim=0)
    alpha_bg = alphas[0]
    alpha_list = [alphas[i + 1] for i in range(len(attn_maps_by_obj))]
    return alpha_bg, alpha_list
