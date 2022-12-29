#### Pathfinder data with extra annotations

This is adapted from the official Pathfinder [repo](https://github.com/drewlinsley/pathfinder). You can use `snakes2_wrapper.py` to generate data. E.g. `python snakes2_wrapper.py 0 10` generates 10 samples with batch id 0. 

To generate 256 x 256 data used in DLR [paper](https://arxiv.org/abs/2212.00768) you can uncomment the following part in `snakes2_wrapper.py` 
```python
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
# args.contour_path = dataset_root  # os.path.join(dataset_root, f'curv_contour_length_{args.contour_length}')
# args.paddle_length = 8
# args.mark_distractors = True
# snakes2.from_wrapper(args)
```  
and then run `bash.sh` containing
```bash
for i in {0..39}
do
    python snakes2_wrapper.py $i 2500 &
done
```
This launches 40 workers with ids 0..39 and assumes you have at least 40 cpu cores. As the data generation for each worker is seeded by its id, the generated data might differ if you change the number of launched workers.

This will save the data in `./data/pathfinder256_segmentation/` and you can manually move this directory to `../../../../data/pathfinder_segmentation/` for the experiments.

Similarly, you can generate data for 128 x 128 and 512 x 512 cases by uncommenting the respective settings in `snakes2_wrapper.py` and launching 40 workers for generating 2500 samples each.