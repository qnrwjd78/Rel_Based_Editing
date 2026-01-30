import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms

from utils import ptp_utils


def show_mask(mask, save_name):
    mask_vis = torch.zeros(64, 64)
    mask_vis[mask] = 1
    transforms.ToPILImage()(mask_vis.unsqueeze(0)).save(save_name)

def build_cross_attention_image(prompt: str,
                                attention_map,
                                tokenizer,
                                token_idx,
                                orig_image=None,
                                caption=None,
                                bbox=None):

    
    # 시각화 이미지를 메모리에서 생성해서 반환
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode

    if caption is None:
        caption = decoder(int(tokens[token_idx]))

    image = attention_map
    image = show_image_relevance(image, orig_image)
    image = image.astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((256, 256)))
    
    # bbox는 원본 해상도 기준 좌표이므로 시각화 크기에 맞게 스케일링해서 표시
    if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        if orig_image is not None:
            src_w, src_h = orig_image.size
        else:
            src_w = src_h = 512
        dst_h, dst_w = image.shape[:2]
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        x1, y1, x2, y2 = bbox
        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))
        x1 = max(0, min(x1, dst_w - 1))
        y1 = max(0, min(y1, dst_h - 1))
        x2 = max(0, min(x2, dst_w - 1))
        y2 = max(0, min(y2, dst_h - 1))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    image = ptp_utils.text_under_image(image, caption)

    result = ptp_utils.view_images(np.stack([image], axis=0))
    return result

def _scale_bbox_for_image(bbox, src_size, dst_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    x1, y1, x2, y2 = bbox
    x1 = int(round(x1 * scale_x))
    y1 = int(round(y1 * scale_y))
    x2 = int(round(x2 * scale_x))
    y2 = int(round(y2 * scale_y))
    x1 = max(0, min(x1, dst_w - 1))
    y1 = max(0, min(y1, dst_h - 1))
    x2 = max(0, min(x2, dst_w - 1))
    y2 = max(0, min(y2, dst_h - 1))
    return x1, y1, x2, y2

def show_image_relevance_diff(image_relevance, image: Image.Image, relevnace_res=16):
    def show_cam_on_image(img, heatmap):
        cam = heatmap + img
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image) / 255.0

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda()
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu()
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    max_abs = image_relevance.abs().max().item()
    if max_abs == 0:
        max_abs = 1.0
    diff = (image_relevance / max_abs).numpy()
    diff = np.clip(diff, -1.0, 1.0)
    t_pos = np.clip(diff, 0.0, 1.0)
    t_neg = np.clip(-diff, 0.0, 1.0)

    heatmap = np.ones((diff.shape[0], diff.shape[1], 3), dtype=np.float32)
    heatmap[..., 1] -= t_pos
    heatmap[..., 2] -= t_pos
    heatmap[..., 0] -= t_neg
    heatmap[..., 1] -= t_neg

    vis = show_cam_on_image(image, heatmap)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def build_diff_attention_image(attention_diff,
                               orig_image=None,
                               caption=None,
                               bbox_s=None,
                               bbox_o=None):
    image = show_image_relevance_diff(attention_diff, orig_image)
    image = image.astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((256, 256)))

    if orig_image is not None:
        src_size = orig_image.size
    else:
        src_size = (512, 512)
    dst_size = (image.shape[1], image.shape[0])
    if bbox_s is not None and isinstance(bbox_s, (list, tuple)) and len(bbox_s) == 4:
        x1, y1, x2, y2 = _scale_bbox_for_image(bbox_s, src_size, dst_size)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if bbox_o is not None and isinstance(bbox_o, (list, tuple)) and len(bbox_o) == 4:
        x1, y1, x2, y2 = _scale_bbox_for_image(bbox_o, src_size, dst_size)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if caption:
        image = ptp_utils.text_under_image(image, caption)

    result = ptp_utils.view_images(np.stack([image], axis=0))
    return result

def show_cross_attention(prompt: str,
                         attention_map,
                         tokenizer,
                         token_idx,
                         orig_image=None,
                         res_fname: str=None):
    result = build_cross_attention_image(
        prompt=prompt,
        attention_map=attention_map,
        tokenizer=tokenizer,
        token_idx=token_idx,
        orig_image=orig_image,
    )
    # 파일 저장은 필요할 때만 수행
    if res_fname:
        result.save(res_fname)
    return result


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
