
from typing import List, Optional
import json
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import logging

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from PIL import Image
from tqdm import tqdm
from utils.vis_utils import (
    show_cross_attention,
    show_mask,
    build_cross_attention_image,
    build_diff_attention_image,
    get_image_grid,
)
import torchvision.transforms as transforms
from utils.ptp_utils import register_time, load_source_latents_t
from utils.loss_utils import (
    attn_inp_loss,
    inpaint_loss,
)
from utils.dift_sd import SDFeaturizer
from utils.ptp_utils import register_cross_attention_control_efficient


logger = logging.get_logger(__name__)

class MnAInpPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds



    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def load_img(self, image_path):
        image_pil = transforms.Resize((512, 512))(Image.open(image_path).convert("RGB"))
        self.orig_img = image_pil
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        image = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image
    
    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    

    @torch.no_grad()
    def dilate_mask(self, mask, config):
        img = torch.zeros(64, 64, dtype=torch.float32)
        img[mask.cpu()] = 1
        img = img.numpy()
        kernel = np.ones((config.dilate_kernel_size, config.dilate_kernel_size), np.uint8)
        dilate = cv2.dilate(img, kernel, 1)

        mask_edge = torch.from_numpy(dilate).to(mask.device)
        mask_edge = mask_edge > 0.5
        mask_edge[mask] = False

        return mask_edge

    @staticmethod
    def _scale_bbox(bbox, src=512, dst=64):
        scale = dst / src
        x1 = int(max(min(round(bbox[0] * scale), dst - 1), 0))
        y1 = int(max(min(round(bbox[1] * scale), dst - 1), 0))
        x2 = int(min(round(bbox[2] * scale), dst))
        y2 = int(min(round(bbox[3] * scale), dst))
        x2 = max(x2, x1 + 1, 0)
        y2 = max(y2, y1 + 1, 0)
        return [x1, y1, x2, y2]

    @staticmethod
    def _bbox_to_mask(bbox, device):
        x1, y1, x2, y2 = bbox
        mask = torch.zeros(64, 64, device=device, dtype=torch.bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _select_attn_map(attn_map, token_indices):
        if attn_map is None or not token_indices:
            return None
        max_idx = attn_map.shape[0]
        valid = [idx for idx in token_indices if 0 <= idx < max_idx]
        if not valid:
            return None
        return attn_map[valid].mean(dim=0)

    def _auto_token_idx(self, prompt, attn_map, mask_box, used_indices=None):
        if attn_map is None or mask_box.sum().item() == 0:
            return []
        used_indices = used_indices or set()
        token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0].tolist()
        invalid_ids = {
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        scores = attn_map[:, mask_box].mean(dim=1)
        best_idx = None
        best_score = None
        for i, token_id in enumerate(token_ids):
            if token_id in invalid_ids or i in used_indices:
                continue
            token_text = self.tokenizer.decode([token_id]).strip()
            if not token_text:
                continue
            score = scores[i].item()
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
        return [best_idx] if best_idx is not None else []

    def _load_mask_image(self, mask_path, device):
        if not mask_path or not os.path.exists(mask_path):
            return None
        try:
            img = Image.open(mask_path).convert("L")
        except Exception:
            return None
        img = img.resize((64, 64), resample=Image.NEAREST)
        arr = np.array(img)
        mask = arr > 127
        return torch.from_numpy(mask).to(device=device, dtype=torch.bool)

    def _resolve_mask_path(self, mask_name, config):
        if not mask_name:
            return None
        if os.path.isabs(mask_name) and os.path.exists(mask_name):
            return mask_name
        if os.path.exists(mask_name):
            return mask_name
        img_path = getattr(config, "img_path", None)
        fname = getattr(config, "fname", None)
        if not img_path and fname:
            img_path = os.path.join("./data", fname)
        if img_path:
            base_dir = os.path.dirname(img_path)
        else:
            base_dir = "."
        candidate = os.path.join(base_dir, mask_name)
        return candidate if os.path.exists(candidate) else None

    @staticmethod
    def _update_latent(latents, loss, step_size):
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond

        return latents

    def _latent_update(
        self,
        dift,
        latent,
        t,
        dift_prompt,
        config,
        token_idx_s,
        token_idx_o,
        token_idx_a,
        mask_s,
        mask_o,
        mask_src_s,
        mask_src_o,
        mask_edge_s,
        mask_edge_o,
        mask_bg,
        erase_mask,
        prefix,
    ):
        scale_range = np.linspace(config.scale_range[0], config.scale_range[1], config.update_steps)

        latent = latent.clone().detach().requires_grad_(True)

        dift.pipe.controller.reset()
        source_ft = dift.forward(
            latent,
            prompt=dift_prompt,
            t=t.item(),
            up_ft_indices=config.up_ft_indices,
            ensemble_size=config.ensemble_size,
        )
        source_ft = source_ft.reshape(source_ft.shape[0], -1)

        dift.pipe.controller.reset()
        dift.pipe.unet.zero_grad()

        attn_map_s_frames = []
        attn_map_a_frames = []
        attn_map_o_frames = []
        attn_map_os_diff_frames = []

        loss_steps = []
        loss_in_s_vals = []
        loss_out_s_vals = []
        loss_in_o_vals = []
        loss_out_o_vals = []
        loss_ipt_s_vals = []
        loss_ipt_o_vals = []
        loss_bg_vals = []
        loss_total_vals = []
        step = 0

        print("updating latent code...")
        for scale in scale_range:
            step += 1
            dift.pipe.controller.reset()
            ft = dift.forward(
                latent,
                prompt=dift_prompt,
                t=t.item(),
                up_ft_indices=config.up_ft_indices,
                ensemble_size=config.ensemble_size,
            )

            ft = ft.reshape(ft.shape[0], -1)

            dift.pipe.controller.merge_attention()
            attn_map = dift.pipe.controller.merge_attn_map
            dift.pipe.controller.reset()
            dift.pipe.unet.zero_grad()

            attn_map_s = self._select_attn_map(attn_map, token_idx_s)
            attn_map_o = self._select_attn_map(attn_map, token_idx_o)
            attn_map_a = self._select_attn_map(attn_map, token_idx_a)

            if attn_map_s is not None and attn_map_o is not None:
                with torch.no_grad():
                    diff_frame = build_diff_attention_image(
                        attention_diff=(attn_map_o - attn_map_s).detach(),
                        orig_image=self.orig_img,
                        caption=f"step {step} O-S",
                        bbox_s=config.bbox_s,
                        bbox_o=config.bbox_o,
                    )
                attn_map_os_diff_frames.append(diff_frame)

            attn_map_s = attn_map_s.reshape(-1) if attn_map_s is not None else None
            attn_map_o = attn_map_o.reshape(-1) if attn_map_o is not None else None
            attn_map_a = attn_map_a.reshape(-1) if attn_map_a is not None else None

            loss_inp_s, loss_out_s = attn_inp_loss(
                attn_map_s, mask_s.reshape(-1)
            )
            loss_inp_o, loss_out_o = attn_inp_loss(
                attn_map_o, mask_o.reshape(-1)
            )
            loss_inp = loss_inp_s + loss_inp_o

            loss_ipt_s = inpaint_loss(ft, source_ft, mask_src_s, mask_edge_s)
            loss_ipt_o = inpaint_loss(ft, source_ft, mask_src_o, mask_edge_o)
            loss_sai = loss_ipt_s + loss_ipt_o

            if mask_bg.sum() > 0:
                loss_bg = torch.nn.SmoothL1Loss()(ft[:, mask_bg], source_ft[:, mask_bg])
            else:
                loss_bg = ft.new_tensor(0.0)

            if erase_mask is not None:
                loss = 0.4 * loss_inp + 0.4 * loss_sai + 0.2 * loss_bg
                latent_updated = self._update_latent(
                    latents=latent,
                    loss=loss,
                    step_size=config.scale_factor * np.sqrt(scale),
                )
                latent = latent * (1.0 - erase_mask) + latent_updated * erase_mask
            else:
                loss = 0.4 * loss_inp + 0.4 * loss_sai + 0.2 * loss_bg
                latent = self._update_latent(
                    latents=latent,
                    loss=loss,
                    step_size=config.scale_factor * np.sqrt(scale),
                )


        if attn_map_os_diff_frames:
            grid = get_image_grid(attn_map_os_diff_frames)
            grid.save(config.output_path / f"{prefix}attn_map_o_minus_s_scales.png")

        print("update completed!")
        return latent.detach()

    def idx_single(self, token):
        if isinstance(token, (list, tuple)):
            if token:
                idx_single = token[0]
        elif isinstance(token, int):
            idx_single = token
        return idx_single

    def _token_texts(self, prompt, token_indices):
        if not token_indices:
            return []
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids[0].tolist()
        tokens = []
        for idx in token_indices:
            if 0 <= idx < len(input_ids):
                text = self.tokenizer.decode([input_ids[idx]]).strip()
                if text:
                    tokens.append(text)
        return tokens

    def _caption_for_token(self, step, label, word, prompt, token_indices):
        token_text = "+".join(self._token_texts(prompt, token_indices)) or "?"
        word_text = word or "?"
        return f"step {step}"

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, config):

        dift = SDFeaturizer()

        scale_range = np.linspace(config.scale_range[0], config.scale_range[1], config.update_steps)

        timesteps = reversed(self.scheduler.timesteps)
        register_cross_attention_control_efficient(dift.pipe)

        dift_prompt = config.edit_prompt
        cond_name = getattr(config, "cond_name", "")
        prefix = f"{cond_name}_" if cond_name else ""
        token_idx_s = []
        token_idx_o = []
        token_idx_a = []

        # 관계용 mask
        mask_src_s = None
        mask_src_o = None
        mask_edge_s = None
        mask_edge_o = None

        # 배경용 mask
        mask_bg = None
        mask_s_name = getattr(config, "mask_s", None)
        mask_o_name = getattr(config, "mask_o", None)
        if not (mask_s_name or mask_o_name):
            cond_path = getattr(config, "cond_path", None)
            if cond_path and os.path.exists(cond_path):
                try:
                    with open(cond_path, "r", encoding="utf8") as fp:
                        cond_data = json.load(fp)
                    mask_s_name = cond_data.get("mask_s")
                    mask_o_name = cond_data.get("mask_o")
                    if not getattr(config, "fname", None) and cond_data.get("fname"):
                        config.fname = cond_data.get("fname")
                except Exception:
                    mask_s_name = None
                    mask_o_name = None

        src_noise_s = None
        src_noise_o = None
        s_step = config.s_step if config.s_step is not None else config.transfer_step
        o_step = config.o_step if config.o_step is not None else config.transfer_step
        print(f"s_step: {s_step}, o_step: {o_step}")
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps, desc="Inversion")): # t: 1000 to 0, t: timestep index
                register_time(self, t.item())

                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                if i == 0:
                    dift.pipe.controller.reset()
                    _ = dift.forward(
                        latent,
                        prompt=dift_prompt,
                        t=t.item(),
                        up_ft_indices=config.up_ft_indices,
                        ensemble_size=config.ensemble_size,
                    )
                    # attn_map [64, 64] 크기 (16*16 해상도에서 평균(up,down 평균)을 내고 resize)
                    # 토큰별 attn_map
                    dift.pipe.controller.merge_attention()
                    attn_map = dift.pipe.controller.merge_attn_map
                    dift.pipe.controller.reset()

                    device = latent.device

                    bbox_s = self._scale_bbox(config.bbox_s)
                    bbox_o = self._scale_bbox(config.bbox_o)

                    def normalize_token_idx(value):
                        print(f"Normalizing token index value: {value}")
                        if isinstance(value, int):
                            return [value]
                        if isinstance(value, list):
                            return value
                        return []

                    # 토큰 설정
                    token_idx_s = normalize_token_idx(getattr(config, "token_idx_s", []))
                    token_idx_o = normalize_token_idx(getattr(config, "token_idx_o", []))
                    token_idx_a = normalize_token_idx(getattr(config, "token_idx_a", []))

                    print(f"Predefined token indices - S: {token_idx_s}, O: {token_idx_o}, A: {token_idx_a}")
                    attn_map_s = self._select_attn_map(attn_map, token_idx_s)
                    attn_map_o = self._select_attn_map(attn_map, token_idx_o)

                    token_idx_o_single = self.idx_single(token_idx_o)
                    token_idx_s_single = self.idx_single(token_idx_s)

                    if attn_map_o is not None and token_idx_o_single is not None:
                        res_fname = str(config.output_path / f"{prefix}attn_map_o.png")
                        with torch.no_grad():
                            show_cross_attention(
                                prompt=dift_prompt,
                                attention_map=attn_map_o.detach(),
                                tokenizer=self.tokenizer,
                                token_idx=token_idx_o_single,
                                orig_image=self.orig_img,
                                res_fname=res_fname,
                            )

                    if attn_map_s is not None and token_idx_s_single is not None:
                        res_fname = str(config.output_path / f"{prefix}attn_map_s.png")
                        with torch.no_grad():
                            show_cross_attention(
                                prompt=dift_prompt,
                                attention_map=attn_map_s.detach(),
                                tokenizer=self.tokenizer,
                                token_idx=token_idx_s_single,
                                orig_image=self.orig_img,
                                res_fname=res_fname,
                            )

                    mask_s_path = self._resolve_mask_path(mask_s_name, config)
                    mask_o_path = self._resolve_mask_path(mask_o_name, config)

                    mask_s = self._load_mask_image(mask_s_path, device) if mask_s_path else None
                    mask_o = self._load_mask_image(mask_o_path, device) if mask_o_path else None

                    if mask_s is None or mask_o is None:
                        raise ValueError(
                            "mask_s/mask_o image is required for inpainting mode. "
                            "Check condition file and mask image paths."
                        )

                    mask_edge_s = self.dilate_mask(mask_s, config)
                    mask_edge_o = self.dilate_mask(mask_o, config)

                    mask_src_s = mask_s.clone()
                    mask_src_o = mask_o.clone()

                    mask_bg = ~(mask_s | mask_o)
                    mask_src_s = mask_src_s.reshape(-1)
                    mask_src_o = mask_src_o.reshape(-1)
                    mask_edge_s = mask_edge_s.reshape(-1)
                    mask_edge_o = mask_edge_o.reshape(-1)
                    mask_bg = mask_bg.reshape(-1)

                if i == s_step and mask_s is not None:
                    mask_s_2d = mask_s.reshape(64, 64).float()
                    src_noise_s = (latent * mask_s_2d.unsqueeze(0).unsqueeze(0)).detach().clone()
                if i == o_step and mask_o is not None:
                    mask_o_2d = mask_o.reshape(64, 64).float()
                    src_noise_o = (latent * mask_o_2d.unsqueeze(0).unsqueeze(0)).detach().clone()

                if i == config.transfer_step:
                    target_bbox_s = None
                    target_bbox_o = None
                    erase_mask = None
                    if mask_s is not None:
                        show_mask(mask_s.detach().cpu(), str(config.output_path / f"{prefix}mask_s.png"))
                    if mask_o is not None:
                        show_mask(mask_o.detach().cpu(), str(config.output_path / f"{prefix}mask_o.png"))

                    target_bbox_s = self._scale_bbox(config.bbox_s)
                    target_bbox_o = self._scale_bbox(config.bbox_o)

                    mask_src_s_2d = mask_src_s.reshape(64, 64) if mask_src_s is not None else None
                    mask_src_o_2d = mask_src_o.reshape(64, 64) if mask_src_o is not None else None
                    
                    if mask_src_s_2d is not None and mask_src_o_2d is not None:
                        print("Using combined erase mask for both subject and object.") 
                        erase_mask = (mask_src_s_2d | mask_src_o_2d).unsqueeze(0).unsqueeze(0).float()

                    with torch.enable_grad():
                        latent = self._latent_update(
                            dift=dift,
                            latent=latent,
                            t=t,
                            dift_prompt=dift_prompt,
                            config=config,
                            token_idx_s=token_idx_s,
                            token_idx_o=token_idx_o,
                            token_idx_a=token_idx_a,
                            mask_s=mask_s,
                            mask_o=mask_o,
                            mask_src_s=mask_src_s,
                            mask_src_o=mask_src_o,
                            mask_edge_s=mask_edge_s,
                            mask_edge_o=mask_edge_o,
                            mask_bg=mask_bg,
                            erase_mask=erase_mask,
                            prefix=prefix,
                        )

                    # if src_noise_s is not None or src_noise_o is not None:
                    #     with torch.no_grad():
                    #         if src_noise_s is not None and target_bbox_s is not None:
                    #             h_s = target_bbox_s[3] - target_bbox_s[1]
                    #             w_s = target_bbox_s[2] - target_bbox_s[0]
                    #             if h_s > 0 and w_s > 0:
                    #                 patch_s = torch.nn.functional.interpolate(
                    #                     src_noise_s, size=(h_s, w_s), mode="bilinear", align_corners=False
                    #                 )
                    #                 mask_s_patch = torch.nn.functional.interpolate(
                    #                     mask_s.reshape(1, 1, 64, 64).float(), size=(h_s, w_s), mode="nearest"
                    #                 )
                    #                 region = latent[:, :, target_bbox_s[1]:target_bbox_s[3], target_bbox_s[0]:target_bbox_s[2]]
                    #                 latent[:, :, target_bbox_s[1]:target_bbox_s[3], target_bbox_s[0]:target_bbox_s[2]] = (
                    #                     region * (1.0 - mask_s_patch) + patch_s * mask_s_patch
                    #                 )
                    #         if src_noise_o is not None and target_bbox_o is not None:
                    #             h_o = target_bbox_o[3] - target_bbox_o[1]
                    #             w_o = target_bbox_o[2] - target_bbox_o[0]
                    #             if h_o > 0 and w_o > 0:
                    #                 patch_o = torch.nn.functional.interpolate(
                    #                     src_noise_o, size=(h_o, w_o), mode="bilinear", align_corners=False
                    #                 )
                    #                 mask_o_patch = torch.nn.functional.interpolate(
                    #                     mask_o.reshape(1, 1, 64, 64).float(), size=(h_o, w_o), mode="nearest"
                    #                 )
                    #                 region = latent[:, :, target_bbox_o[1]:target_bbox_o[3], target_bbox_o[0]:target_bbox_o[2]]
                    #                 latent[:, :, target_bbox_o[1]:target_bbox_o[3], target_bbox_o[0]:target_bbox_o[2]] = (
                    #                     region * (1.0 - mask_o_patch) + patch_o * mask_o_patch
                    #                 )


                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps

                torch.save(latent, os.path.join(config.latents_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(config.latents_path, f'noisy_latents_{t}.pt'))

        return latent

    @torch.no_grad()
    def edit(
            self,
            guidance_scale: float = 7.5,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            config = None
    ):

        timesteps = self.scheduler.timesteps

        with torch.autocast(device_type='cuda', dtype=torch.float32):

            for i, t in enumerate(tqdm(timesteps, desc="Editing  ")):
                
                # register timesteps
                register_time(self, t.item())

                pnp_guidance_embeds = self.get_text_embeds(config.edit_prompt, "").chunk(2)[0]
        
                # expand the latents if we are doing classifier free guidance
                source_latents = load_source_latents_t(t, config.latents_path)
                latent_model_input = torch.cat([source_latents] + ([latents] * 2))


                text_embed_input = torch.cat([pnp_guidance_embeds, prompt_embeds], dim=0)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embed_input
                ).sample

                # perform guidance
                _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                              
        return latents
