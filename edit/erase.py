import torch


def erase_loss_optim_latent(x_t_src, mask_src_lat, loss_fn, step_size=0.1, steps=5, **loss_kwargs):
    """
    입력:
      x_t_src: t* latent [B,4,H,W]
      mask_src_lat: [B,1,H,W]
      loss_fn: latent -> loss (스칼라)
    출력:
      x_t_bg: t* latent
    """
    latent = x_t_src.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss = loss_fn(latent, **loss_kwargs)
        grad = torch.autograd.grad(loss, [latent])[0]
        latent = latent - step_size * grad
        # 마스크 영역만 업데이트
        latent = x_t_src * (1.0 - mask_src_lat) + latent * mask_src_lat
        latent = latent.detach().requires_grad_(True)
    return latent.detach()


# erase_loss_global_optim (latent 입력/출력)
# bg 전체 영역을 loss에 사용하도록 loss_fn에 mask_bg 전달

def erase_loss_global_optim_latent(
    x_t_src,
    mask_src_lat,
    loss_fn,
    step_size=0.1,
    steps=5,
    **loss_kwargs,
):
    """
    입력:
      x_t_src: t* latent [B,4,H,W]
      mask_src_lat: [B,1,H,W]
      loss_fn: (latent, mask_bg, **kwargs) -> loss
    출력:
      x_t_bg: t* latent
    """
    mask_bg = (1.0 - mask_src_lat).clamp(0, 1)
    latent = x_t_src.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss = loss_fn(latent, mask_bg, **loss_kwargs)
        grad = torch.autograd.grad(loss, [latent])[0]
        latent = latent - step_size * grad
        latent = x_t_src * (1.0 - mask_src_lat) + latent * mask_src_lat
        latent = latent.detach().requires_grad_(True)
    return latent.detach()


# erase_masked_renoise (정확 의미: x0_bg 기반)
# 소스 객체 마스크 영역만 더 높은 t의 노이즈 상태로 치환

def erase_masked_renoise_from_x0(
    x0_bg,
    mask_src_lat,
    scheduler,
    t_star,
    t_high,
    eps_star=None,
    eps_high=None,
):
    """
    입력:
      x0_bg: clean bg latent [B,4,H,W]
      mask_src_lat: [B,1,H,W]
      t_star: scheduler.timesteps[idx_star]
      t_high: t_star보다 큰 timestep 값
    출력:
      x_t_bg: t* latent
    """
    if eps_star is None:
        eps_star = torch.randn_like(x0_bg)
    if eps_high is None:
        eps_high = torch.randn_like(x0_bg)

    x_t_star = scheduler.add_noise(x0_bg, eps_star, t_star)
    x_t_high = scheduler.add_noise(x0_bg, eps_high, t_high)

    return x_t_star * (1.0 - mask_src_lat) + x_t_high * mask_src_lat


def erase_masked_random_noise_from_x0(
    x0_bg,
    mask_src_lat,
    scheduler,
    t_star,
    t_high=None,
    eps_star=None,
    eps_high=None,
):
    """
    입력:
      x0_bg: clean bg latent [B,4,H,W]
      mask_src_lat: [B,1,H,W]
      t_star: scheduler.timesteps[idx_star]
    출력:
      x_t_bg: t* latent

    배경은 x0_bg에서 t_star로 노이징, 마스크 영역은 순수 노이즈로 채움.
    """
    if eps_star is None:
        eps_star = torch.randn_like(x0_bg)
    x_t_star = scheduler.add_noise(x0_bg, eps_star, t_star)

    eps_rand = torch.randn_like(x0_bg)
    x_t_rand = scheduler.add_noise(torch.zeros_like(x0_bg), eps_rand, t_star)

    return x_t_star * (1.0 - mask_src_lat) + x_t_rand * mask_src_lat


# erase_inpaint_then_inv_latent
# 마스크 영역을 배경 평균값(라텐트)으로 채움

def erase_inpaint_then_inv_latent(x_t_src, mask_src_lat):
    """
    입력:
      x_t_src: t* latent [B,4,H,W]
      mask_src_lat: [B,1,H,W]
    출력:
      x_t_bg: t* latent
    """
    masked = x_t_src * (1.0 - mask_src_lat)
    denom = (1.0 - mask_src_lat).sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = masked.sum(dim=(2, 3), keepdim=True) / denom
    return x_t_src * (1.0 - mask_src_lat) + mean * mask_src_lat
