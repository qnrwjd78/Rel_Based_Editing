python dataset/trans.py --input data
set/hoi/hoi_generation

python preprocess.py  --model_root ./model  --model_cfg model/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml  --checkpoint model/sam2/checkpoints/sam2.1_hiera_large.pt  --output_dir ./data/hoi_mask  --condition_file ./condition/hoi_generation/gen_0001.json