# Move and Act: Enhanced Object Manipulation and Background Integrity for Image Editing

This repository provides the code to edit an object's action in an image and control where the edited object is generated.

## Setup
Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate ldm

pip install --force-reinstall --no-deps \
  diffusers==0.18.2 \
  huggingface_hub==0.16.4 \
  accelerate==0.20.3
```

### CUDA 12.1
If you need CUDA 12.1, use:

```bash
conda env create -f environment_121.yaml
conda activate ldm

pip install --force-reinstall --no-deps \
  diffusers==0.18.2 \
  huggingface_hub==0.16.4 \
  accelerate==0.20.3
```

## Usage
Edit the action of an object and specify the generation position:

```bash
python run_mna.py --img_path './data/0040.jpg' --cond_path './condition/0040.json'
```

Generate the edited object at its original location:

```bash
python run_mna.py --img_path './data/0012.jpg' --cond_path './condition/0012.json'
```

## Outputs
Results are written under `outputs/` with the following structure:

```text
outputs/
|-- text prompt/
|   |-- 42.png
|   |-- 42_bbox.png
|   |-- 1.png
|   |-- 1_bbox.png
|   |-- ...
```

## Project Layout
Commonly used directories:

- `condition/`: JSON condition files
- `data/`: input images
- `outputs/`: generated results
- `model/`: external model code and assets
