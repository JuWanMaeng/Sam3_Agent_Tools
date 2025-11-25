import torch
from PIL import Image
import numpy as np
# sam3 라이브러리 가정
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- SAM-3 모델 초기화 및 추론 (기존 코드) ---
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
# 원본 이미지를 RGB로 변환하여 로드
image_path = r"assets\images\test_image.jpg"
image = Image.open(image_path).convert("RGB")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="long hair girls")

masks = output.get('masks', None)
# ---------------------------------------------

# 마스크 오버레이 로직
if masks is None or (isinstance(masks, torch.Tensor) and masks.numel() == 0):
    print("No masks detected (masks is None or empty tensor).")
else:
    print(f"DEBUG: Detected {len(masks)} masks. Overlaying all masks.")

    # 1. 모든 마스크를 합치는 과정 (Boolean OR 연산)
    
    # 마스크 크기 확인 (가장 처음 마스크 기준으로)
    if isinstance(masks, torch.Tensor):
        first_mask_tensor = masks[0]
    elif isinstance(masks, list):
        first_mask_tensor = masks[0]

    # 텐서를 NumPy 배열로 변환
    if hasattr(first_mask_tensor, "cpu"):
        mask_shape = first_mask_tensor.cpu().numpy().shape
    else:
        mask_shape = np.array(first_mask_tensor).shape

    # 마스크 배열이 (H, W) 형태인지 확인하고 (1, H, W) 형태라면 (H, W)로 변환
    if len(mask_shape) == 3 and mask_shape[0] == 1:
        H, W = mask_shape[1], mask_shape[2]
    elif len(mask_shape) == 2:
        H, W = mask_shape[0], mask_shape[1]
    else:
        print("Error: Mask shape is unexpected.")
        exit()

    # 모든 마스크를 합칠 빈 배열 (이진 마스크) 초기화
    combined_mask_np = np.zeros((H, W), dtype=bool)

    # 모든 마스크 반복 처리
    for mask_tensor in masks:
        # 텐서를 NumPy 배열로 변환
        if hasattr(mask_tensor, "cpu"):
            mask_np = mask_tensor.cpu().numpy()
        else:
            mask_np = np.array(mask_tensor)

        # (1, H, W) 형태라면 (H, W)로 압축
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np.squeeze(0)

        # 이진 마스크로 변환 (True/False)
        binary_mask = (mask_np > 0)
        
        # 논리합 (OR) 연산으로 마스크 합치기
        combined_mask_np = np.logical_or(combined_mask_np, binary_mask)

    # 2. 합쳐진 마스크를 RGBA 오버레이 이미지로 변환
    
    # 오버레이 색상 (반투명 빨간색: Red=255, Green=0, Blue=0, Alpha=150)
    color = (255, 0, 0, 150) 
    
    # RGBA 이미지 배열 초기화 (완전 투명)
    mask_img_array = np.zeros((H, W, 4), dtype=np.uint8)
    
    # 마스크가 True인 위치에만 색상 및 투명도 적용
    # np.where는 True/False 조건에 따라 값을 설정
    mask_img_array[combined_mask_np] = color
    
    mask_overlay = Image.fromarray(mask_img_array, 'RGBA')
    
    # 3. 원본 이미지와 오버레이 합성
    
    # 원본 이미지를 RGBA로 변환
    base_img_rgba = image.convert("RGBA")
    
    # 마스크 이미지를 원본 이미지 크기에 맞게 크기 조정 (크기가 다른 경우)
    if mask_overlay.size != base_img_rgba.size:
        # Image.Resampling.NEAREST를 사용하여 마스크의 픽셀 경계를 유지
        mask_overlay = mask_overlay.resize(base_img_rgba.size, Image.Resampling.NEAREST)

    # 두 이미지 합성
    result_img = Image.alpha_composite(base_img_rgba, mask_overlay)
    
    # RGB로 다시 변환하여 저장
    result_img.convert("RGB").save("all_masks_overlay.jpg") 
    print("Saved all_masks_overlay.jpg with all detected masks combined and overlaid.")