# controller 초기화 및 forward pass 진행
dift.pipe.controller.reset()
_ = dift.forward(
    latent.detach(),
    prompt=dift_prompt,
    t=t.item(),
    up_ft_indices=config.up_ft_indices,
    ensemble_size=config.ensemble_size,
)
dift.pipe.controller.merge_attention()

# merge된 attnetion map 가져오기
attn_map_step = dift.pipe.controller.merge_attn_map


# source token에 대한 attention map 선택
attn_s = self._select_attn_map(attn_map_step, token_idx_s)

# attention map 시각화
idx_s_single = self.idx_single(token_idx_s)
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

# 시각화 저장
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

@staticmethod
def _select_attn_map(attn_map, token_indices):
    if attn_map is None or not token_indices:
        return None
    max_idx = attn_map.shape[0]
    valid = [idx for idx in token_indices if 0 <= idx < max_idx]
    if not valid:
        return None
    return attn_map[valid].mean(dim=0)