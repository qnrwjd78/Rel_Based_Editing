# conda ldm_inp 생성/설치 기록

아래 순서대로 실행해서 `ldm_inp` 환경을 만들고 SAM2를 설치했어.

1) 새 conda 환경 생성 (Python 3.10)
```
conda create -n ldm_inp python=3.10 -y
```

2) PyTorch + torchvision 설치 (CUDA 12.1)
```
conda run -n ldm_inp pip install --index-url https://download.pytorch.org/whl/cu121 \\
  torch==2.5.1 torchvision==0.20.1
```

3) SAM2 설치 (editable, CUDA 확장 빌드 비활성)
```
conda run -n ldm_inp env SAM2_BUILD_CUDA=0 \\
  python -m pip install -e \"/home/work/sjchoi/SceneGraph/move-act/model/sam2[notebooks]\"
```

4) 설치 확인
```
conda run -n ldm_inp python -c \"import torch, sam2; print(torch.__version__); print(sam2.__name__)\"
```
