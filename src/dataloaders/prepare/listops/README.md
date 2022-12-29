#### ListOps data with extra annotations

`listops.py` is adapted from the Long Range Arena `ListOps` [script](https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py).

Usage:
```bash
# for recreating LRA data
python listops.py --max_length 2000 --min_length=500 --output_dir ./data/listops/ 

# OR

# for recreating ListOpsSubTrees data with similar input lengths
python listops.py --output_dir ./data/listops_subtrees/ --num_train_samples 96000 --num_valid_samples 2000 --num_test_samples 2000 --no_paranthesis --unique_templates --max_length 2000 --min_length=1500 --task subtrees --num_workers 8

# OR
    
# for recreating ListOpsSubTrees data with longer input lengths as done in the DLR paper
python listops.py --output_dir ./data/listops_subtrees/ --num_train_samples 96000 --num_valid_samples 2000 --num_test_samples 2000 --no_paranthesis --unique_templates --max_length 8180 --min_length=7000 --task subtrees --max_args 9 --max_depth 15 --num_workers 30
```

This will save the data in `./data/listops_subtrees/` and you can manually move it to `../../../../data/` for the experiments. The last setting uses a significant amount of RAM.

**Notes**:  
    1.  use `--no_paranthesis` flag to make stored data shorter as the listops tokenizer in the LRA pipeline anyways removes paranthesis and hence using this flag will NOT affect the input to your model.  
    2.  values corresponding to all sub-nodes of the tree are also saved and can optionally be used as extra supervision.  
    3.  `--num_workers` controls then number of cpu cores used. Each worker is seeded by its ID so *changing the number of workers will change the generated data*. 

