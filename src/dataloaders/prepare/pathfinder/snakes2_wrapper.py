import os
os.environ['OPENBLAS_NUM_THREADS']='1'  # fix for error: PyCapsule_Import could not import module "datetime"
import time
import sys
import numpy as np


import snakes2


class Args:
    def __init__(
        self,
        contour_path = './data', 
        batch_id=0, 
        n_images = 10,
        window_size=[256,256], 
        padding=1, 
        antialias_scale = 4,
        LABEL=1, 
        seed_distance= 27, 
        marker_radius = 3,
        contour_length=15, 
        distractor_length=5, 
        num_distractor_snakes=6, 
        snake_contrast_list=[1.], 
        use_single_paddles=False,
        mark_distractors=False,
        max_target_contour_retrial = 4, 
        max_distractor_contour_retrial = 4, 
        max_paddle_retrial=2,
        continuity = 1.4, 
        paddle_length=5, 
        paddle_thickness=1.5, 
        paddle_margin_list=[4], 
        paddle_contrast_list=[1.],
        pause_display=False, 
        save_images=True, 
        save_metadata=True,
        save_main_segments=True,
        seed=0,
    ):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.padding = padding
        self.antialias_scale = antialias_scale

        self.LABEL = LABEL
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_snakes = num_distractor_snakes
        self.snake_contrast_list = snake_contrast_list
        self.use_single_paddles = use_single_paddles
        self.mark_distractors = mark_distractors
        
        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list  # if multiple elements in a list, a number will be sampled in each IMAGE
        self.paddle_contrast_list = paddle_contrast_list  # if multiple elements in a list, a number will be sampled in each PADDLE

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata
        self.save_main_segments = save_main_segments
        self.seed = seed

args = Args()

args.use_single_paddles = False
args.segmentation_task = False
args.segmentation_task_double_circle = False


if len(sys.argv) > 1:
    args.batch_id = int(sys.argv[1])
    args.n_images = int(sys.argv[2])


# ----- 128 x 128 -----
# dataset_root = './data/pathfinder128_segmentation/'
# args.padding = 1
# args.paddle_margin_list = [2,3]   # gap between dashes
# args.seed_distance = 22           # distance between main segments - ambiguous if too small
# args.window_size = [128,128]
# args.marker_radius = 2.1
# args.contour_length = 14
# args.paddle_thickness = 1.5       # dash thickness
# args.antialias_scale = 2
# args.continuity = 1.8             # spread of paths - if small, paths curve quickly
# args.paddle_length = 5            # dash len
# args.distractor_length = args.contour_length // 3   
# args.num_distractor_snakes = 64 / args.distractor_length
# args.snake_contrast_list = [0.9]  # how much darker are dashes compared to markers
# args.contour_path = dataset_root  
# args.mark_distractors = True
# snakes2.from_wrapper(args)



# -------- 256 x 256 --------
# dataset_root = './data/pathfinder256_segmentation/'
# args.padding = 1
# args.paddle_margin_list = [4]
# args.window_size = [256,256]
# args.marker_radius = 4.2
# args.contour_length = 16
# args.paddle_thickness = 2
# args.antialias_scale = 2
# args.continuity = 2.2
# args.distractor_length = args.contour_length // 3
# args.num_distractor_snakes = 16
# args.snake_contrast_list = [1.0]
# args.contour_path = dataset_root  
# args.paddle_length = 8
# args.mark_distractors = True
# snakes2.from_wrapper(args)



# -------- 512 x 512 --------
# dataset_root = './data/pathfinder512_segmentation/'
# args.padding = 1
# args.seed_distance = 40
# args.paddle_margin_list = [7] 
# args.window_size = [512,512]
# args.marker_radius = 9
# args.contour_length = 12
# args.paddle_thickness = 6
# args.antialias_scale = 3
# args.continuity = 2.5
# args.distractor_length = args.contour_length // 3
# args.num_distractor_snakes = 18
# args.snake_contrast_list = [1.0]  
# args.contour_path = dataset_root  
# args.paddle_length = 22
# args.mark_distractors = True
# snakes2.from_wrapper(args)
# 40 hrs per 2500 per cpu



"""
Example usage:
python snakes2_wrapper.py 0 10 
"""