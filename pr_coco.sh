# -*- coding: utf-8 -*-
# If you want to train vae yourself
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_vae_coco.py --image_folder /vision/7052107/Dalle/coco/train2017 --anno_folder /vision/7052107/Dalle/coco/annotations/captions_train2017.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_dalle.py --vae_path ./vae-final_coco.pt --image_folder /vision/7052107/Dalle/coco/train2017 --anno_folder /vision/7052107/Dalle/coco/annotations/captions_train2017.json


# CUDA_VISIBLE_DEVICES=6,7 python train_dalle.py --dalle_path ./dalle-final.pt --image_text_folder /vision/7052107/Dalle/2D-Shape-Generator/output7/
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'rect is red' --num_images 4 --outputs_dir ./outputs/out9
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'ellipse is green' --num_images 4 --outputs_dir ./outputs/out9
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'star5 is blue' --num_images 4 --outputs_dir ./outputs/out9
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'star8 is white' --num_images 4 --outputs_dir ./outputs/out9
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'poly3 is magenta' --num_images 4 --outputs_dir ./outputs/out9
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'poly5 is yellow' --num_images 4 --outputs_dir ./outputs/out9
# CUDA_VISIBLE_DEVICES=6 python generate.py --dalle_path ./dalle-final.pt --text 'poly7 is cyan' --num_images 4 --outputs_dir ./outputs/out9

# train set
# CUDA_VISIBLE_DEVICES=5 python generate.py --dalle_path ./dalle-final.pt --text 'ellipse is white' --num_images 4 --outputs_dir ./outputs/out8




# If you want to use DALLE's published vae
# CUDA_VISIBLE_DEVICES=0 python train_dalle.py --image_text_folder /vision/7016118/ori_codes/2D-Shape-Generator/output2/
# CUDA_VISIBLE_DEVICES=0 python generate.py --dalle_path ./dalle.pt   --text 'rect is red' --num_images 4 --outputs_dir ./outputs/wholedataㄴ
