import torch


def attn_in_out_loss(attn_map, mask_box, topk_ratio):
    zero = torch.tensor(0.0, device=mask_box.device)
    if attn_map is None or mask_box.sum().item() == 0:
        return zero, zero
    values_in = attn_map[mask_box]
    values_out = attn_map[~mask_box]
    if values_in.numel() == 0:
        loss_in = zero
    else:
        k = max(1, int(values_in.numel() * topk_ratio))
        k = min(k, values_in.numel())
        loss_in = 1.0 - values_in.topk(k).values.mean()
    loss_out = values_out.mean() if values_out.numel() > 0 else zero
    return loss_in, loss_out


def attn_inp_loss(attn_map, mask):
    zero = torch.tensor(0.0, device=mask.device)
    if attn_map is None or mask.sum().item() == 0:
        return zero, zero
    values_in = attn_map[mask]
    values_out = attn_map[~mask]
    if values_in.numel() == 0:
        loss_in = zero
    else:
        loss_in = values_in.mean()
    loss_out = values_out.mean() if values_out.numel() > 0 else zero
    return loss_in, loss_out


def inpaint_loss(ft, source_ft, mask_src, mask_edge):
    mask_src_count = int(mask_src.sum().item())
    mask_edge_count = int(mask_edge.sum().item())
    if mask_src_count == 0 or mask_edge_count == 0:
        return ft.new_tensor(0.0)
    ft_edge = source_ft[:, mask_edge]
    if ft_edge.numel() == 0:
        return ft.new_tensor(0.0)
    fts_edge = ft_edge
    while fts_edge.shape[1] < mask_src_count:
        fts_edge = torch.cat([fts_edge, ft_edge], dim=1)
    return torch.nn.SmoothL1Loss()(ft[:, mask_src], fts_edge[:, :mask_src_count])


def dice_loss(attn_map, mask_box, eps=1e-8):
    if attn_map is None or mask_box.sum().item() == 0:
        return attn_map.new_tensor(0.0) if attn_map is not None else torch.tensor(0.0, device=mask_box.device)
    a_min = attn_map.min()
    a_max = attn_map.max()
    a_norm = (attn_map - a_min) / (a_max - a_min + eps)
    m = mask_box.float()
    inter = (a_norm * m).sum()
    denom = a_norm.sum() + m.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def kl_loss(attn_map, mask_box, tau=1.0, eps=1e-8):
    if attn_map is None or mask_box.sum().item() == 0:
        return attn_map.new_tensor(0.0) if attn_map is not None else torch.tensor(0.0, device=mask_box.device)
    q = torch.nn.functional.softmax(attn_map / tau, dim=0)
    p = mask_box.float()
    p = p / (p.sum() + eps)
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)))


def _bg_loss(ft, source_ft, mask_bg):
    if mask_bg is None:
        return ft.new_tensor(0.0)
    if mask_bg.sum() > 0:
        return torch.nn.SmoothL1Loss()(ft[:, mask_bg], source_ft[:, mask_bg])
    return ft.new_tensor(0.0)


def compute_relation_loss(
    *,
    attn_map_s,
    attn_map_o,
    attn_map_a,
    mask_box_s_flat,
    mask_box_o_flat,
    mask_box_a_flat,
    ft,
    source_ft,
    mask_src_s,
    mask_src_o,
    mask_edge_s,
    mask_edge_o,
    mask_bg,
    topk_ratio,
):
    l_dice_s = dice_loss(attn_map_s, mask_box_s_flat)
    l_dice_o = dice_loss(attn_map_o, mask_box_o_flat)
    l_dice = 0.5 * (l_dice_s + l_dice_o)

    l_kl_s = kl_loss(attn_map_s, mask_box_s_flat)
    l_kl_o = kl_loss(attn_map_o, mask_box_o_flat)
    l_kl = 0.5 * (l_kl_s + l_kl_o)

    loss_in_s, loss_out_s = attn_in_out_loss(attn_map_s, mask_box_s_flat, topk_ratio)
    loss_in_o, loss_out_o = attn_in_out_loss(attn_map_o, mask_box_o_flat, topk_ratio)
    loss_in_a, loss_out_a = attn_in_out_loss(attn_map_a, mask_box_a_flat, topk_ratio)

    loss_oii = loss_in_s + 3 * loss_out_s + loss_in_o + 3 * loss_out_o

    loss_ipt_s = inpaint_loss(ft, source_ft, mask_src_s, mask_edge_s)
    loss_ipt_o = inpaint_loss(ft, source_ft, mask_src_o, mask_edge_o)
    loss_sai = loss_ipt_s + loss_ipt_o

    loss_bg = _bg_loss(ft, source_ft, mask_bg)

    loss = 0.1 * loss_bg + 0.0 * loss_sai + 0.3 * loss_oii + 0.3 * l_dice + 0.3 * l_kl

    metrics = {
        "loss_in_s": loss_in_s,
        "loss_out_s": loss_out_s,
        "loss_in_o": loss_in_o,
        "loss_out_o": loss_out_o,
        "loss_in_a": loss_in_a,
        "loss_out_a": loss_out_a,
        "loss_ipt_s": loss_ipt_s,
        "loss_ipt_o": loss_ipt_o,
        "loss_bg": loss_bg,
        "loss_sai": loss_sai,
        "loss_oii": loss_oii,
        "loss_dice": l_dice,
        "loss_kl": l_kl,
    }
    return loss, metrics


def compute_single_loss(
    *,
    attn_map_main,
    mask_box_flat,
    ft,
    source_ft,
    mask_src,
    mask_edge,
    mask_bg,
    topk_ratio,
):
    loss_attn, loss_zero = attn_in_out_loss(attn_map_main, mask_box_flat, topk_ratio)
    loss_oii = loss_attn + loss_zero

    loss_ipt = inpaint_loss(ft, source_ft, mask_src, mask_edge)
    loss_bg = _bg_loss(ft, source_ft, mask_bg)

    loss = 0.1 * loss_bg + 0.0 * loss_ipt + 0.3 * loss_oii

    metrics = {
        "loss_attn": loss_attn,
        "loss_zero": loss_zero,
        "loss_ipt": loss_ipt,
        "loss_bg": loss_bg,
        "loss_oii": loss_oii,
    }
    return loss, metrics
