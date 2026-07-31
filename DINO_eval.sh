coco_path="/media/mldadmin/home/s125mdg35_05/dataset/data"
checkpoint="/media/mldadmin/home/s125mdg35_05/DINO/DINO/logs/DINO/R50-MS4/checkpoint0015.pth"
export CUDA_VISIBLE_DEVICES="1" && python main.py \
  --output_dir logs/DINO/R50-MS4-eval \
	-c config/DINO/DINO_4scale_swin_eval.py --coco_path $coco_path  \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 batch_size=5
