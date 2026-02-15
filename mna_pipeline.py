
from typing import List, Optional
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
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
from utils.loss_utils import compute_relation_loss, compute_single_loss
from utils.latent_update import update_latent, _update_latent
from edit.overlap import overlap_soft_zbuffer
import torchvision.transforms as transforms
from utils.ptp_utils import register_time, load_source_latents_t
from utils.dift_sd import SDFeaturizer
from utils.ptp_utils import register_cross_attention_control_efficient


logger = logging.get_logger(__name__)

class MnAPipeline(StableDiffusionPipeline):
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

    def _load_mask_64(self, mask_path: str, device, *, dst_img=512, dst_lat=64) -> Optional[torch.Tensor]:
        """
        Load a binary segmentation mask image and convert it to a bool mask on the 64x64 latent grid.
        Expected mask format: white(>0) = foreground, black = background.
        """
        if not mask_path or not os.path.exists(mask_path):
            return None
        m = Image.open(mask_path).convert("L")
        # Match the resized image size (512x512) used by the pipeline, then downsample to latent grid.
        m = m.resize((dst_img, dst_img), resample=Image.NEAREST)
        m = m.resize((dst_lat, dst_lat), resample=Image.NEAREST)
        arr = (np.array(m) > 0)
        return torch.from_numpy(arr).to(device=device, dtype=torch.bool)

    @staticmethod
    def _grid_with_cols(images: List[Image.Image], cols: int) -> Image.Image:
        if not images:
            return Image.new("RGB", (1, 1), (255, 255, 255))
        cols = max(1, int(cols or 1))
        cols = min(cols, len(images))
        rows = int(math.ceil(len(images) / cols))
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for idx, img in enumerate(images):
            x = (idx % cols) * w
            y = (idx // cols) * h
            grid.paste(img, (x, y))
        return grid
    
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

        use_relation = getattr(config, "use_relation", False)
        dift_prompt = config.edit_prompt if use_relation else (config.inv_prompt or config.edit_prompt)
        cond_name = getattr(config, "cond_name", "")
        prefix = f"{cond_name}_" if cond_name else ""

        token_idx_main = []
        token_idx_s = []
        token_idx_o = []
        token_idx_a = []

        # 단일 Object 용 mask
        mask_box_flat = None
        mask_box = None
        mask_src = None
        mask_edge = None

        # 관계용 mask
        mask_box_s = None
        mask_box_o = None
        mask_box_a = None
        mask_box_s_flat = None
        mask_box_o_flat = None
        mask_box_a_flat = None
        mask_src_s = None
        mask_src_o = None
        # Keep 64x64 (2d) versions for erase losses that need spatial indexing.
        mask_src_s_2d = None
        mask_src_o_2d = None
        mask_edge_s = None
        mask_edge_o = None

        # 배경용 mask
        mask_bg = None

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            save_step_grids = bool(getattr(config, "save_step_attn_grids", False)) and use_relation
            inv_s_imgs: List[Image.Image] = []
            inv_o_imgs: List[Image.Image] = []

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

                    if use_relation:
                        bbox_s = self._scale_bbox(config.bbox_s)
                        bbox_o = self._scale_bbox(config.bbox_o)
                        if config.bbox_a:
                            bbox_a_raw = config.bbox_a
                        else:
                            xs = sorted([config.bbox_s[0], config.bbox_s[2], config.bbox_o[0], config.bbox_o[2]])
                            ys = sorted([config.bbox_s[1], config.bbox_s[3], config.bbox_o[1], config.bbox_o[3]])
                            bbox_a_raw = [xs[1], ys[1], xs[2], ys[2]]
                        bbox_a = self._scale_bbox(bbox_a_raw)

                        mask_box_s = self._bbox_to_mask(bbox_s, device)
                        mask_box_o = self._bbox_to_mask(bbox_o, device)
                        mask_box_a = self._bbox_to_mask(bbox_a, device)

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

                        if attn_map_s is not None:
                            mask_s = attn_map_s >= 0.01
                        else:
                            bbox_s_src = config.bbox_s_src if config.bbox_s_src else config.bbox_s
                            mask_s = self._bbox_to_mask(self._scale_bbox(bbox_s_src), device)

                        if attn_map_o is not None:
                            mask_o = attn_map_o >= 0.01
                        else:
                            bbox_o_src = config.bbox_o_src if config.bbox_o_src else config.bbox_o
                            mask_o = self._bbox_to_mask(self._scale_bbox(bbox_o_src), device)

                        mask_edge_s = self.dilate_mask(mask_s, config)
                        mask_edge_o = self.dilate_mask(mask_o, config)

                        # If segmentation masks are provided (e.g. SAM2), prefer them as the source erase masks.
                        # These masks already represent the source object regions precisely, so we do not
                        # subtract target boxes (which can under-erase when target overlaps source).
                        seg_s = self._load_mask_64(getattr(config, "mask_s_path", None), device)
                        seg_o = self._load_mask_64(getattr(config, "mask_o_path", None), device)

                        # Subject
                        if seg_s is not None:
                            mask_s = seg_s
                            mask_edge_s = self.dilate_mask(mask_s, config)
                            mask_src_s = mask_s.clone()
                        else:
                            mask_src_s = mask_s.clone()

                        # Object
                        if seg_o is not None:
                            mask_o = seg_o
                            mask_edge_o = self.dilate_mask(mask_o, config)
                            mask_src_o = mask_o.clone()
                        else:
                            mask_src_o = mask_o.clone()

                        mask_bg = ~(mask_src_s | mask_src_o | mask_box_a)

                        # Preserve 2D masks before flattening for loss functions.
                        mask_src_s_2d = mask_src_s.clone()
                        mask_src_o_2d = mask_src_o.clone()

                        mask_box_s_flat = mask_box_s.reshape(-1)
                        mask_box_o_flat = mask_box_o.reshape(-1)
                        mask_box_a_flat = mask_box_a.reshape(-1)
                        mask_src_s = mask_src_s.reshape(-1)
                        mask_src_o = mask_src_o.reshape(-1)
                        mask_edge_s = mask_edge_s.reshape(-1)
                        mask_edge_o = mask_edge_o.reshape(-1)
                        mask_bg = mask_bg.reshape(-1)
                    
                    else:
                        bbox = self._scale_bbox(config.bbox)
                        mask_box = self._bbox_to_mask(bbox, device)
                        token_idx_main = [config.inv_token_idx]
                        attn_map_main = self._select_attn_map(attn_map, token_idx_main)
                        if attn_map_main is not None:
                            mask = attn_map_main >= 0.01
                        else:
                            mask = mask_box.clone()

                        mask_edge = self.dilate_mask(mask, config)
                        mask_src = mask.clone()
                        mask_src[mask_box] = False
                        mask_bg = ~(mask_src | mask_box)

                        mask_box_flat = mask_box.reshape(-1)
                        mask_src = mask_src.reshape(-1)
                        mask_edge = mask_edge.reshape(-1)
                        mask_bg = mask_bg.reshape(-1)

                # Per-step attention (inversion). This is computed on the *current* latent at the *current* timestep.
                # Note: timesteps are iterated as `reversed(self.scheduler.timesteps)` in this repo.
                if save_step_grids and token_idx_s and token_idx_o:
                    try:
                        dift.pipe.controller.reset()
                        _ = dift.forward(
                            latent.detach(),
                            prompt=dift_prompt,
                            t=t.item(),
                            up_ft_indices=config.up_ft_indices,
                            ensemble_size=config.ensemble_size,
                        )
                        dift.pipe.controller.merge_attention()
                        attn_map_step = dift.pipe.controller.merge_attn_map
                    finally:
                        dift.pipe.controller.reset()

                    attn_s = self._select_attn_map(attn_map_step, token_idx_s)
                    attn_o = self._select_attn_map(attn_map_step, token_idx_o)
                    idx_s_single = self.idx_single(token_idx_s)
                    idx_o_single = self.idx_single(token_idx_o)
                    if attn_s is not None and idx_s_single is not None:
                        inv_s_imgs.append(
                            build_cross_attention_image(
                                prompt=dift_prompt,
                                attention_map=attn_s.detach(),
                                tokenizer=self.tokenizer,
                                token_idx=idx_s_single,
                                orig_image=self.orig_img,
                                caption=f"S i={i} t={int(t)}",
                            )
                        )
                    if attn_o is not None and idx_o_single is not None:
                        inv_o_imgs.append(
                            build_cross_attention_image(
                                prompt=dift_prompt,
                                attention_map=attn_o.detach(),
                                tokenizer=self.tokenizer,
                                token_idx=idx_o_single,
                                orig_image=self.orig_img,
                                caption=f"O i={i} t={int(t)}",
                            )
                        )


                if i == config.transfer_step:
                    # 기존 Move&Act 최적화(for scale in scale_range)는 제거하고,
                    # transfer_step에서 erase/compose를 딱 1번만 적용한다.
                    erase_method = getattr(config, 'erase_method', 'none')
                    compose_method = getattr(config, 'compose_method', 'none')
                    if erase_method != 'none' or compose_method != 'none':
                        erase_kwargs = dict(getattr(config, 'erase_kwargs', {}) or {})
                        compose_kwargs = dict(getattr(config, 'compose_kwargs', {}) or {})
                        b, c, h, w = latent.shape

                        # Build object patches for compose *before* erase mutates the latent.
                        # This enables move/swap behavior: extract from source region and paste into target boxes.
                        latent_pre_erase = latent.detach()

                        # Save attention maps for subject/object at three stages:
                        # pre-erase, post-erase, post-compose.
                        def _save_stage_attn(latent_stage, stage_name: str):
                            if not use_relation:
                                return
                            if not token_idx_s or not token_idx_o:
                                return
                            try:
                                dift.pipe.controller.reset()
                                _ = dift.forward(
                                    latent_stage,
                                    prompt=dift_prompt,
                                    t=t.item(),
                                    up_ft_indices=config.up_ft_indices,
                                    ensemble_size=config.ensemble_size,
                                )
                                dift.pipe.controller.merge_attention()
                                attn_map_stage = dift.pipe.controller.merge_attn_map
                                print(f"Shape of attn_map_step: {attn_map_step.shape}")
                            finally:
                                dift.pipe.controller.reset()

                            attn_map_s_stage = self._select_attn_map(attn_map_stage, token_idx_s)

                            attn_map_o_stage = self._select_attn_map(attn_map_stage, token_idx_o)
                            token_idx_s_single = self.idx_single(token_idx_s)
                            token_idx_o_single = self.idx_single(token_idx_o)

                            if attn_map_s_stage is not None and token_idx_s_single is not None:
                                res_fname = str(config.output_path / f"{prefix}attn_map_s_{stage_name}.png")
                                with torch.no_grad():
                                    show_cross_attention(
                                        prompt=dift_prompt,
                                        attention_map=attn_map_s_stage.detach(),
                                        tokenizer=self.tokenizer,
                                        token_idx=token_idx_s_single,
                                        orig_image=self.orig_img,
                                        res_fname=res_fname,
                                    )
                            if attn_map_o_stage is not None and token_idx_o_single is not None:
                                res_fname = str(config.output_path / f"{prefix}attn_map_o_{stage_name}.png")
                                with torch.no_grad():
                                    show_cross_attention(
                                        prompt=dift_prompt,
                                        attention_map=attn_map_o_stage.detach(),
                                        tokenizer=self.tokenizer,
                                        token_idx=token_idx_o_single,
                                        orig_image=self.orig_img,
                                        res_fname=res_fname,
                                    )

                        _save_stage_attn(latent_pre_erase, "pre_erase")

                        mask_src_lat = (mask_src_s | mask_src_o).reshape(1, 1, h, w).float()

                        # Target masks come from target bboxes (not source masks).
                        tgt_bbox_s = self._scale_bbox(config.bbox_s)
                        tgt_bbox_o = self._scale_bbox(config.bbox_o)
                        mask_tgt_s_lat = self._bbox_to_mask(tgt_bbox_s, latent.device).reshape(1, 1, h, w).float()
                        mask_tgt_o_lat = self._bbox_to_mask(tgt_bbox_o, latent.device).reshape(1, 1, h, w).float()

                        masks_for_objs = [mask_tgt_s_lat, mask_tgt_o_lat]

                        # Prepare paste patches for compose (relation mode only).
                        objs = []
                        if compose_method != 'none' and use_relation:
                            import torch.nn.functional as F

                            def _bbox_from_mask64(mask64: torch.Tensor):
                                if mask64 is None:
                                    return None
                                if mask64.ndim != 2:
                                    mask64 = mask64.reshape(64, 64)
                                ys, xs = torch.where(mask64)
                                if ys.numel() == 0:
                                    return None
                                y1 = int(ys.min().item())
                                y2 = int(ys.max().item()) + 1
                                x1 = int(xs.min().item())
                                x2 = int(xs.max().item()) + 1
                                return [x1, y1, x2, y2]

                            def _make_patch_from_bbox(src_bbox, tgt_bbox, tgt_mask_lat):
                                x1s, y1s, x2s, y2s = src_bbox
                                x1t, y1t, x2t, y2t = tgt_bbox
                                hs = max(1, y2s - y1s)
                                ws = max(1, x2s - x1s)
                                ht = max(1, y2t - y1t)
                                wt = max(1, x2t - x1t)
                                src = latent_pre_erase[:, :, y1s:y2s, x1s:x2s]
                                if src.shape[-2:] != (ht, wt):
                                    src = F.interpolate(src, size=(ht, wt), mode="bilinear", align_corners=False)
                                out = torch.zeros_like(latent_pre_erase)
                                out[:, :, y1t:y2t, x1t:x2t] = src
                                # Keep patch content only where the target mask is active (important for gaussian/weighted compose).
                                return out * tgt_mask_lat

                            def _make_patch_from_mask(src_mask64: torch.Tensor, tgt_bbox):
                                """
                                Extract a masked crop from the *source* region using a segmentation mask,
                                compute a tight bbox, then paste it into the *target* bbox.

                                Returns:
                                  xt_patch_full: [B,4,H,W] patch placed in target bbox
                                  mask_tgt_lat: [B,1,H,W] alpha mask placed in target bbox (tight, not full bbox)
                                """
                                if src_mask64 is None:
                                    return None, None
                                if src_mask64.ndim != 2:
                                    src_mask64 = src_mask64.reshape(64, 64)
                                ys, xs = torch.where(src_mask64)
                                if ys.numel() == 0:
                                    return None, None
                                y1s = int(ys.min().item())
                                y2s = int(ys.max().item()) + 1
                                x1s = int(xs.min().item())
                                x2s = int(xs.max().item()) + 1
                                x1t, y1t, x2t, y2t = tgt_bbox
                                ht = max(1, y2t - y1t)
                                wt = max(1, x2t - x1t)

                                src = latent_pre_erase[:, :, y1s:y2s, x1s:x2s]
                                src_m = src_mask64[y1s:y2s, x1s:x2s].float().unsqueeze(0).unsqueeze(0)
                                src = src * src_m

                                if src.shape[-2:] != (ht, wt):
                                    src = F.interpolate(src, size=(ht, wt), mode="bilinear", align_corners=False)
                                    src_m = F.interpolate(src_m, size=(ht, wt), mode="nearest")

                                out = torch.zeros_like(latent_pre_erase)
                                out[:, :, y1t:y2t, x1t:x2t] = src

                                mask_tgt = torch.zeros((1, 1, latent_pre_erase.shape[-2], latent_pre_erase.shape[-1]), device=latent_pre_erase.device)
                                mask_tgt[:, :, y1t:y2t, x1t:x2t] = src_m
                                return out, mask_tgt

                            # Source bboxes: prefer explicit *_src, otherwise derive from source masks.
                            src_bbox_s = self._scale_bbox(config.bbox_s_src) if getattr(config, "bbox_s_src", None) else _bbox_from_mask64(mask_s)
                            src_bbox_o = self._scale_bbox(config.bbox_o_src) if getattr(config, "bbox_o_src", None) else _bbox_from_mask64(mask_o)

                            if src_bbox_s is None or src_bbox_o is None:
                                # If we cannot find source regions, skip compose (will behave like erase-only).
                                objs = []
                            else:
                                # Subject -> subject target box
                                o_s = type("Obj", (), {})()
                                # Prefer SAM2/segmentation masks when available to avoid bbox-only artifacts.
                                patch_s, tgt_alpha_s = _make_patch_from_mask(mask_s, tgt_bbox_s)
                                if patch_s is None:
                                    o_s.mask_tgt_lat = mask_tgt_s_lat
                                    o_s.xt_patch_full = _make_patch_from_bbox(src_bbox_s, tgt_bbox_s, mask_tgt_s_lat)
                                else:
                                    o_s.mask_tgt_lat = tgt_alpha_s
                                    o_s.xt_patch_full = patch_s
                                o_s.priority = 0.0
                                objs.append(o_s)

                                # Object -> object target box
                                o_o = type("Obj", (), {})()
                                patch_o, tgt_alpha_o = _make_patch_from_mask(mask_o, tgt_bbox_o)
                                if patch_o is None:
                                    o_o.mask_tgt_lat = mask_tgt_o_lat
                                    o_o.xt_patch_full = _make_patch_from_bbox(src_bbox_o, tgt_bbox_o, mask_tgt_o_lat)
                                else:
                                    o_o.mask_tgt_lat = tgt_alpha_o
                                    o_o.xt_patch_full = patch_o
                                o_o.priority = 0.0
                                objs.append(o_o)
                        # 1) erase 1회 적용
                        if erase_method != 'none':
                            if erase_method in ('loss_optim', 'loss_global', 'inpaint_then_inv'):
                                erase_kwargs.setdefault('mask_src_lat', mask_src_lat)

                            # Provide a default erase loss for loss_optim/loss_global:
                            # minimize subject/object cross-attention inside the *source* masks.
                            if erase_method in ('loss_optim', 'loss_global') and 'loss_fn' not in erase_kwargs:
                                w_s = float(getattr(config, "erase_w_s", 1.0))
                                w_o = float(getattr(config, "erase_w_o", 1.0))

                                src_s = mask_src_s_2d  # 64x64 bool
                                src_o = mask_src_o_2d  # 64x64 bool
                                token_s = list(token_idx_s) if token_idx_s else []
                                token_o = list(token_idx_o) if token_idx_o else []
                                up_idx = list(getattr(config, "up_ft_indices", [2]))
                                ens = int(getattr(config, "ensemble_size", 1) or 1)

                                def _erase_loss_fn(latent_in, mask_bg=None, **_):
                                    dift.pipe.controller.reset()
                                    _ = dift.forward(
                                        latent_in,
                                        prompt=dift_prompt,
                                        t=t.item(),
                                        up_ft_indices=up_idx,
                                        ensemble_size=ens,
                                    )
                                    dift.pipe.controller.merge_attention()
                                    attn_map_cur = dift.pipe.controller.merge_attn_map
                                    dift.pipe.controller.reset()

                                    attn_s_cur = self._select_attn_map(attn_map_cur, token_s)
                                    attn_o_cur = self._select_attn_map(attn_map_cur, token_o)
                                    loss = latent_in.new_tensor(0.0)
                                    if attn_s_cur is not None and src_s is not None and src_s.any():
                                        loss = loss + w_s * attn_s_cur[src_s].mean()
                                    if attn_o_cur is not None and src_o is not None and src_o.any():
                                        loss = loss + w_o * attn_o_cur[src_o].mean()
                                    return loss

                                erase_kwargs['loss_fn'] = _erase_loss_fn




                            if erase_method in ('masked_renoise', 'masked_random_noise'):



                                # x0_bg 추정: x_t = mu*x0 + sigma*eps  => x0 = (x_t - sigma*eps)/mu



                                with torch.no_grad():



                                    eps_star = self.unet(latent, t, encoder_hidden_states=cond_batch).sample



                                x0_bg_est = (latent - sigma * eps_star) / mu



                                erase_kwargs.update({



                                    'x0_bg': x0_bg_est.detach(),



                                    'mask_src_lat': mask_src_lat,



                                    'scheduler': self.scheduler,



                                    't_star': t,



                                    't_high': getattr(config, 't_high', t),



                                    'eps_star': eps_star.detach(),



                                })




                            # loss_optim/loss_global require gradients; enable them even though this
                            # function is under @torch.no_grad().
                            if erase_method in ("loss_optim", "loss_global"):
                                with torch.enable_grad():
                                    latent = _update_latent(
                                        latent,
                                        erase=erase_method,
                                        compose=None,
                                        erase_kwargs=erase_kwargs,
                                        strict=False,
                                    )
                            else:
                                latent = _update_latent(
                                    latent,
                                    erase=erase_method,
                                    compose=None,
                                    erase_kwargs=erase_kwargs,
                                    strict=False,
                                )

                        _save_stage_attn(latent.detach(), "post_erase")




                        # 2) compose 1회 적용 (precomputed patches, pasted into target boxes)



                        if compose_method != 'none':
                            if not use_relation:
                                # Non-relation path: keep previous behavior.
                                objs = []
                                for m in masks_for_objs:
                                    o = type('Obj', (), {})()
                                    o.mask_tgt_lat = m
                                    o.xt_patch_full = latent_pre_erase.detach() * m
                                    o.priority = 0.0
                                    objs.append(o)
                            # Relation path: objs already computed above (may be empty if source regions were missing).




                            if compose_method in ('norm', 'norm_overlap_avg', 'norm_feather'):



                                compose_kwargs.update({'objs': objs})



                            elif compose_method == 'gaussian':



                                compose_kwargs.update({'objs': objs, 't': t, 'scheduler': self.scheduler})



                            elif compose_method == 'weighted':



                                alpha_bg, alpha_list = overlap_soft_zbuffer(objs)



                                m_transition = torch.zeros_like(mask_src_lat)



                                compose_kwargs.update({



                                    'objs': objs,



                                    'alpha_list': alpha_list,



                                    'alpha_bg': alpha_bg,



                                    'm_transition': m_transition,



                                    't': t,



                                    'scheduler': self.scheduler,



                                })

                            elif compose_method == 'weighted_transition':
                                # Transition region is computed inside the compose function.
                                compose_kwargs.update({'objs': objs, 't': t, 'scheduler': self.scheduler})




                            latent = _update_latent(



                                latent,



                                erase=None,



                                compose=compose_method,



                                compose_kwargs=compose_kwargs,



                                strict=False,



                            )

                        _save_stage_attn(latent.detach(), "post_compose")



                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps

                torch.save(latent, os.path.join(config.latents_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(config.latents_path, f'noisy_latents_{t}.pt'))

        # Save 4 grids for inversion: (0->transfer) and (transfer->end), for S and O.
        if save_step_grids and inv_s_imgs and inv_o_imgs:
            cols = int(getattr(config, "attn_grid_cols", 10) or 10)
            split = int(getattr(config, "transfer_step", 0) or 0)
            split = max(0, min(split, len(inv_s_imgs) - 1))

            s_0 = inv_s_imgs[: split + 1]
            s_1 = inv_s_imgs[split:]
            o_0 = inv_o_imgs[: split + 1]
            o_1 = inv_o_imgs[split:]

            self._grid_with_cols(s_0, cols).save(str(config.output_path / f"{prefix}attn_grid_s_inv_0_{split}.png"))
            self._grid_with_cols(o_0, cols).save(str(config.output_path / f"{prefix}attn_grid_o_inv_0_{split}.png"))
            self._grid_with_cols(s_1, cols).save(str(config.output_path / f"{prefix}attn_grid_s_inv_{split}_{len(inv_s_imgs)-1}.png"))
            self._grid_with_cols(o_1, cols).save(str(config.output_path / f"{prefix}attn_grid_o_inv_{split}_{len(inv_o_imgs)-1}.png"))

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
        use_relation = bool(getattr(config, "use_relation", False))
        save_step_grids = bool(getattr(config, "save_step_attn_grids", False)) and use_relation
        cond_name = getattr(config, "cond_name", "")
        prefix = f"{cond_name}_" if cond_name else ""
        dift_prompt = config.edit_prompt if use_relation else (config.inv_prompt or config.edit_prompt)

        if save_step_grids:
            dift = SDFeaturizer()
            register_cross_attention_control_efficient(dift.pipe)
            token_idx_s = getattr(config, "token_idx_s", []) or []
            token_idx_o = getattr(config, "token_idx_o", []) or []
            den_s_imgs: List[Image.Image] = []
            den_o_imgs: List[Image.Image] = []

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

                # Per-step attention grids for denoising (50->0 in scheduler order).
                if save_step_grids and token_idx_s and token_idx_o:
                    try:
                        dift.pipe.controller.reset()
                        _ = dift.forward(
                            latents.detach(),
                            prompt=dift_prompt,
                            t=t.item(),
                            up_ft_indices=getattr(config, "up_ft_indices", [2]),
                            ensemble_size=getattr(config, "ensemble_size", 1),
                        )
                        dift.pipe.controller.merge_attention()
                        attn_map_step = dift.pipe.controller.merge_attn_map
                    finally:
                        dift.pipe.controller.reset()

                    attn_s = self._select_attn_map(attn_map_step, token_idx_s)
                    attn_o = self._select_attn_map(attn_map_step, token_idx_o)
                    idx_s_single = self.idx_single(token_idx_s)
                    idx_o_single = self.idx_single(token_idx_o)
                    if attn_s is not None and idx_s_single is not None:
                        den_s_imgs.append(
                            build_cross_attention_image(
                                prompt=dift_prompt,
                                attention_map=attn_s.detach(),
                                tokenizer=self.tokenizer,
                                token_idx=idx_s_single,
                                orig_image=self.orig_img,
                                caption=f"S denoise i={i} t={int(t)}",
                            )
                        )
                    if attn_o is not None and idx_o_single is not None:
                        den_o_imgs.append(
                            build_cross_attention_image(
                                prompt=dift_prompt,
                                attention_map=attn_o.detach(),
                                tokenizer=self.tokenizer,
                                token_idx=idx_o_single,
                                orig_image=self.orig_img,
                                caption=f"O denoise i={i} t={int(t)}",
                            )
                        )
                              
        if save_step_grids and den_s_imgs and den_o_imgs:
            cols = int(getattr(config, "attn_grid_cols", 10) or 10)
            self._grid_with_cols(den_s_imgs, cols).save(str(config.output_path / f"{prefix}attn_grid_s_denoise.png"))
            self._grid_with_cols(den_o_imgs, cols).save(str(config.output_path / f"{prefix}attn_grid_o_denoise.png"))

        return latents
