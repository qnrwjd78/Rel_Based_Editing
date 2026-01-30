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
