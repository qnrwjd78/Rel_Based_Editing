
from typing import List, Optional
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

    @staticmethod
    def _attn_in_out_loss(attn_map, mask_box, topk_ratio):
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

    @staticmethod
    def _inpaint_loss(ft, source_ft, mask_src, mask_edge):
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
    
    @staticmethod
    def _update_latent(latents, loss, step_size):
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond

        return latents

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
        mask_edge_s = None
        mask_edge_o = None

        # 배경용 mask
        mask_bg = None

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

                        mask_src_s = mask_s.clone()
                        mask_src_s[mask_box_s] = False
                        mask_src_o = mask_o.clone()
                        mask_src_o[mask_box_o] = False

                        mask_bg = ~(mask_src_s | mask_src_o | mask_box_a)

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



                if i == config.transfer_step:
                    with torch.enable_grad():

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

                        # 스케일별 attention 이미지를 모아두기 위한 리스트
                        attn_map_s_frames = []
                        attn_map_a_frames = []
                        attn_map_o_frames = []
                        attn_map_os_diff_frames = []
                        
                        # 손실값을 step별로 모아두기 위한 리스트
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

                            if use_relation:
                                attn_map_s = self._select_attn_map(attn_map, token_idx_s)
                                attn_map_o = self._select_attn_map(attn_map, token_idx_o)
                                attn_map_a = self._select_attn_map(attn_map, token_idx_a)

                                token_idx_s_single = self.idx_single(token_idx_s)
                                if attn_map_s is not None and token_idx_s_single is not None:
                                    s_word = getattr(config, "s_word", "")
                                    with torch.no_grad():
                                        frame = build_cross_attention_image(
                                            prompt=dift_prompt,
                                            attention_map=attn_map_s.detach(),
                                            tokenizer=self.tokenizer,
                                            token_idx=token_idx_s_single,
                                            orig_image=self.orig_img,
                                            caption=self._caption_for_token(
                                                step, "S", s_word, dift_prompt, token_idx_s
                                            ),
                                            bbox=config.bbox_s     
                                        )
                                    attn_map_s_frames.append(frame)
                                
                                token_idx_a_single = self.idx_single(token_idx_a)
                                if attn_map_a is not None and token_idx_a_single is not None:
                                    a_word = getattr(config, "a_word", "")
                                    with torch.no_grad():
                                        frame = build_cross_attention_image(
                                            prompt=dift_prompt,
                                            attention_map=attn_map_a.detach(),
                                            tokenizer=self.tokenizer,
                                            token_idx=token_idx_a_single,
                                            orig_image=self.orig_img,
                                            caption=self._caption_for_token(
                                                step, "A", a_word, dift_prompt, token_idx_a
                                            ),
                                            bbox=config.bbox_a     
                                        )
                                    attn_map_a_frames.append(frame)
                                
                                token_idx_o_single = self.idx_single(token_idx_o)
                                if attn_map_o is not None and token_idx_o_single is not None:
                                    o_word = getattr(config, "o_word", "")
                                    # 루프마다 저장하지 않고 이미지로 누적
                                    with torch.no_grad():
                                        frame = build_cross_attention_image(
                                            prompt=dift_prompt,
                                            attention_map=attn_map_o.detach(),
                                            tokenizer=self.tokenizer,
                                            token_idx=token_idx_o_single,
                                            orig_image=self.orig_img,
                                            caption=self._caption_for_token(
                                                step, "O", o_word, dift_prompt, token_idx_o
                                            ),
                                            bbox=config.bbox_o     
                                        )
                                    attn_map_o_frames.append(frame)

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
                                

                                ### loss 계산
                                # loss dice
                                def _dice_loss(attn_map, mask_box, eps=1e-8):
                                    if attn_map is None or mask_box.sum().item() == 0:
                                        return attn_map.new_tensor(0.0) if attn_map is not None else torch.tensor(0.0, device=mask_box.device)
                                    a_min = attn_map.min()
                                    a_max = attn_map.max()
                                    a_norm = (attn_map - a_min) / (a_max - a_min + eps)
                                    m = mask_box.float()
                                    inter = (a_norm * m).sum()
                                    denom = a_norm.sum() + m.sum()
                                    return 1.0 - (2.0 * inter + eps) / (denom + eps)

                                l_dice_s = _dice_loss(attn_map_s, mask_box_s_flat)
                                l_dice_o = _dice_loss(attn_map_o, mask_box_o_flat)
                                l_dice = 0.5 * (l_dice_s + l_dice_o)

                                # loss kl
                                def _kl_loss(attn_map, mask_box, tau=1.0, eps=1e-8):
                                    if attn_map is None or mask_box.sum().item() == 0:
                                        return attn_map.new_tensor(0.0) if attn_map is not None else torch.tensor(0.0, device=mask_box.device)
                                    q = torch.nn.functional.softmax(attn_map / tau, dim=0)
                                    p = mask_box.float()
                                    p = p / (p.sum() + eps)
                                    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)))

                                l_kl_s = _kl_loss(attn_map_s, mask_box_s_flat)
                                l_kl_o = _kl_loss(attn_map_o, mask_box_o_flat)
                                l_kl = 0.5 * (l_kl_s + l_kl_o)

                                # loss oii
                                loss_in_s, loss_out_s = self._attn_in_out_loss(
                                    attn_map_s, mask_box_s_flat, config.P
                                )
                                loss_in_o, loss_out_o = self._attn_in_out_loss(
                                    attn_map_o, mask_box_o_flat, config.P
                                )
                                loss_in_a, loss_out_a = self._attn_in_out_loss(
                                    attn_map_a, mask_box_a_flat, config.P
                                )
                                loss_oii = (
                                    loss_in_s + 3*loss_out_s
                                    + loss_in_o + 3*loss_out_o
                                    # + loss_in_a + 3*loss_out_a
                                )
                                
                                # loss sai
                                loss_ipt_s = self._inpaint_loss(ft, source_ft, mask_src_s, mask_edge_s)
                                loss_ipt_o = self._inpaint_loss(ft, source_ft, mask_src_o, mask_edge_o)
                                loss_sai = loss_ipt_s + loss_ipt_o
                            else:
                                attn_map_main = self._select_attn_map(attn_map, token_idx_main)
                                attn_map_main = attn_map_main.reshape(-1) if attn_map_main is not None else None
                                loss_attn, loss_zero = self._attn_in_out_loss(
                                    attn_map_main, mask_box_flat, config.P
                                )
                                loss_oii = loss_attn + loss_zero
                                loss_sai = self._inpaint_loss(ft, source_ft, mask_src, mask_edge)

                            if mask_bg.sum() > 0:
                                loss_bg = torch.nn.SmoothL1Loss()(ft[:, mask_bg], source_ft[:, mask_bg])
                            else:
                                loss_bg = ft.new_tensor(0.0)

                            loss = 0.1*loss_bg + 0.0*loss_sai + 0.3*loss_oii + 0.3*l_dice + 0.3*l_kl
                            # loss 곡선 기록 (relation 모드에서만)
                            if use_relation:
                                loss_steps.append(step)
                                loss_in_s_vals.append(float(loss_in_s.detach().cpu().item()))
                                loss_out_s_vals.append(float(loss_out_s.detach().cpu().item()))
                                loss_in_o_vals.append(float(loss_in_o.detach().cpu().item()))
                                loss_out_o_vals.append(float(loss_out_o.detach().cpu().item()))
                                loss_ipt_s_vals.append(float(loss_ipt_s.detach().cpu().item()))
                                loss_ipt_o_vals.append(float(loss_ipt_o.detach().cpu().item()))
                                loss_bg_vals.append(float(loss_bg.detach().cpu().item()))
                                loss_total_vals.append(float(loss.detach().cpu().item()))

                            # print('loss_bg: ', loss_bg)
                            # print('loss_ipt: ', loss_ipt)
                            # print('loss_attn: ', loss_attn)
                            # print('loss_zero: ', loss_zero)
                            # print('loss: ', loss)
                            latent = self._update_latent(
                                latents=latent,
                                loss=loss,
                                step_size=config.scale_factor * np.sqrt(scale),
                            )

                        # 누적된 시각화 이미지를 한 장으로 저장
                        # if attn_map_s_frames:
                        #     grid = get_image_grid(attn_map_s_frames)
                        #     grid.save(config.output_path / f"{prefix}attn_map_s_scales.png")
                        # if attn_map_a_frames:
                        #     grid = get_image_grid(attn_map_a_frames)
                        #     grid.save(config.output_path / f"{prefix}attn_map_a_scales.png")
                        # if attn_map_o_frames:
                        #     grid = get_image_grid(attn_map_o_frames)
                        #     grid.save(config.output_path / f"{prefix}attn_map_o_scales.png")
                        # if attn_map_os_diff_frames:
                        #     grid = get_image_grid(attn_map_os_diff_frames)
                        #     grid.save(config.output_path / f"{prefix}attn_map_o_minus_s_scales.png")
                        
                        # 누적된 손실값을 한 그래프에 저장
                        # if loss_steps:
                        #     plt.figure(figsize=(10, 6))
                        #     plt.plot(loss_steps, loss_in_s_vals, label="loss_in_s")
                        #     plt.plot(loss_steps, loss_out_s_vals, label="loss_out_s")
                        #     plt.plot(loss_steps, loss_in_o_vals, label="loss_in_o")
                        #     plt.plot(loss_steps, loss_out_o_vals, label="loss_out_o")
                        #     plt.plot(loss_steps, loss_ipt_s_vals, label="loss_ipt_s")
                        #     plt.plot(loss_steps, loss_ipt_o_vals, label="loss_ipt_o")
                        #     plt.plot(loss_steps, loss_bg_vals, label="loss_bg")
                        #     plt.plot(loss_steps, loss_total_vals, label="loss_total")
                        #     plt.xlabel("step")
                        #     plt.ylabel("loss")
                        #     plt.legend()
                        #     plt.tight_layout()
                        #     plt.savefig(str(config.output_path / f"{prefix}loss_curves.png"))
                        #     plt.close()

                        #     plt.figure(figsize=(10, 6))
                        #     plt.plot(loss_steps, loss_in_s_vals, label="loss_in_s")
                        #     plt.plot(loss_steps, loss_out_s_vals, label="loss_out_s")
                        #     plt.plot(loss_steps, loss_in_o_vals, label="loss_in_o")
                        #     plt.plot(loss_steps, loss_out_o_vals, label="loss_out_o")
                        #     plt.xlabel("step")
                        #     plt.ylabel("loss")
                        #     plt.legend()
                        #     plt.tight_layout()
                        #     plt.savefig(str(config.output_path / f"{prefix}loss_oii_curves.png"))
                        #     plt.close()
                        
                        #     plt.figure(figsize=(10, 6))
                        #     plt.plot(loss_steps, loss_ipt_s_vals, label="loss_ipt_s")
                        #     plt.plot(loss_steps, loss_ipt_o_vals, label="loss_ipt_o")
                        #     plt.xlabel("step")
                        #     plt.ylabel("loss")
                        #     plt.legend()
                        #     plt.tight_layout()
                        #     plt.savefig(str(config.output_path / f"{prefix}loss_sai_curves.png"))
                        #     plt.close()
                        print("update completed!")


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
