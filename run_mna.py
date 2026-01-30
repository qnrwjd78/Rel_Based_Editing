import json
import pyrallis
import torch
from PIL import Image
from config import RunConfig
from mna_pipeline import MnAPipeline
from mna_inpainting_pipeline import MnAInpPipeline
from utils import ptp_utils, vis_utils
from utils.drawer import DashedImageDraw
import numpy as np
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def compute_relation_bbox(bbox_s, bbox_o):
    xs = sorted([bbox_s[0], bbox_s[2], bbox_o[0], bbox_o[2]])
    ys = sorted([bbox_s[1], bbox_s[3], bbox_o[1], bbox_o[3]])
    return [xs[1], ys[1], xs[2], ys[2]]


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "/path/to/sd-v2.1"
        print("stable-diffusion-2-1")
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
        # stable_diffusion_version = "../stable-diffusion-v1-4"
    if config.pipeline == "base":
        stable = MnAPipeline.from_pretrained(stable_diffusion_version).to(device)
        print("MNA Base Pipeline is used.")
    else:
        stable = MnAInpPipeline.from_pretrained(stable_diffusion_version).to(device)
        print("MNA Inpainting Pipeline is used.")

    return stable

def run_on_prompt(model,
                  config: RunConfig) -> Image.Image:

    model.scheduler.set_timesteps(config.n_inference_steps, device=model._execution_device)

    # condition
    cond = model.get_text_embeds(config.edit_prompt, "")
    cond_inv = model.get_text_embeds(config.inv_prompt, "")

    # load and encode image
    image = model.load_img(config.img_path)
    latent = model.encode_imgs(image)

    # inversion
    ptp_utils.no_register(model)
    latents = model.ddim_inversion(cond_inv[1].unsqueeze(0), latent, config)

    # editing
    attn_timesteps = model.scheduler.timesteps[config.attn_steps:] if config.attn_steps >= 0 else []
    ptp_utils.register_attention_control_efficient(model, attn_timesteps)
    edit_latent = model.edit(prompt_embeds=cond, latents=latents, config=config)

    # decode and save image
    recon_image = model.decode_latents(edit_latent)
    image = transforms.ToPILImage()(recon_image[0])
    image.save('./edit.png')

    return image

def build_edit_prompt_from_words(s_word: str, a_word: str, o_word: str) -> str:
    return f"a {s_word} is {a_word} a {o_word}"

def _find_sublist(haystack, needle):
    if not needle:
        return []
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return list(range(i, i + len(needle)))
    return []

def find_token_indices(tokenizer, prompt: str, word: str):
    if not word:
        return []
    tokenized = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids[0].tolist()
    for candidate in (f" {word}", word):
        word_ids = tokenizer.encode(candidate, add_special_tokens=False)
        idxs = _find_sublist(input_ids, word_ids)
        if idxs:
            return idxs
    return []


def main_impl(config: RunConfig):
    stable = load_model(config)

    cond_paths = list(getattr(config, "cond_paths", []) or [])
    cond_names = list(getattr(config, "cond_names", []) or [])
    if cond_paths:
        if cond_names and len(cond_names) != len(cond_paths):
            raise ValueError("cond_names and cond_paths must have the same length.")
        cond_iter = [
            (cond_names[idx] if idx < len(cond_names) else f"cond_{idx + 1}", path)
            for idx, path in enumerate(cond_paths)
        ]
    else:
        cond_iter = [(getattr(config, "cond_name", "cond"), config.cond_path)]

    for cond_name, cond_path in cond_iter:
        config.cond_name = cond_name
        config.cond_path = cond_path
        config.use_relation = False

        with open(config.cond_path, 'r', encoding='utf8') as fp:
            cond = json.load(fp)
            
        if "inv_prompt" in cond:
            config.inv_prompt = cond["inv_prompt"]

        s_word = cond.get("s_word") or cond.get("s")
        a_word = cond.get("a_word") or cond.get("a")
        o_word = cond.get("o_word") or cond.get("o")
        if s_word and a_word and o_word:
            config.s_word = s_word
            config.a_word = a_word
            config.o_word = o_word
            config.edit_prompt = build_edit_prompt_from_words(s_word, a_word, o_word)
            config.token_idx_s = find_token_indices(stable.tokenizer, config.edit_prompt, s_word)
            config.token_idx_a = find_token_indices(stable.tokenizer, config.edit_prompt, a_word)
            config.token_idx_o = find_token_indices(stable.tokenizer, config.edit_prompt, o_word)
            print(f"Edit prompt: {config.edit_prompt}")
            print(f"Identified token indices - S: {config.token_idx_s}, A: {config.token_idx_a}, O: {config.token_idx_o}")

        if "bbox_s" in cond and "bbox_o" in cond:
            config.use_relation = True
            config.bbox_s = cond["bbox_s"]
            config.bbox_o = cond["bbox_o"]
            config.bbox_a = compute_relation_bbox(config.bbox_s, config.bbox_o)
            config.bbox_s_src = cond.get("bbox_s_src")
            config.bbox_o_src = cond.get("bbox_o_src")
            config.bbox = config.bbox_s

        elif "bbox" in cond:
            config.use_relation = False
            config.bbox = cond["bbox"]
            config.bbox_s = config.bbox
            config.bbox_o = config.bbox
            config.bbox_a = config.bbox

        images = []
        for seed in config.seeds:
            ptp_utils.seed_everything(seed)
            print(f"Current seed is : {seed}")
            
            image = run_on_prompt(model=stable,
                                  config=config)
            prompt_output_path = config.output_path / config.edit_prompt[:100]
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'{seed}.png')
            images.append(image)

            canvas = Image.fromarray(np.zeros((image.size[0], image.size[0], 3), dtype=np.uint8) + 220)
            draw = DashedImageDraw(canvas)

            
            if config.use_relation:
                x1, y1, x2, y2 = config.bbox_s
                draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[0], width=5)
                x1, y1, x2, y2 = config.bbox_o
                draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[1], width=5)
            else:
                x1, y1, x2, y2 = config.bbox
                draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[2], width=5)
            canvas.save(prompt_output_path / f'{seed}_bbox.png')

        # save a grid of results across all seeds
        joined_image = vis_utils.get_image_grid(images)
        joined_image.save(config.output_path / f'{config.edit_prompt}.png')


@pyrallis.wrap()
def main(config: RunConfig):
    main_impl(config)


if __name__ == '__main__':
    main()
