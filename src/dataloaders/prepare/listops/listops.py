"""
Adapted from https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py
Default args are same as Long Range Arena.

Usage:
    1. python listops.py --max_length 2000 --min_length=500 --output_dir ./data/listops/   # recreates LRA data 
    
    OR
    
    2. python listops.py --output_dir ./data/listops_subtrees/ --num_train_samples 96000 --num_valid_samples 2000 --num_test_samples 2000 --no_paranthesis --unique_templates --max_length 2000 --min_length=1500 --task subtrees --num_workers 8

    OR
    
    3. python listops.py --output_dir ./data/listops_subtrees/ --num_train_samples 96000 --num_valid_samples 2000 --num_test_samples 2000 --no_paranthesis --unique_templates --max_length 8180 --min_length=7000 --task subtrees --max_args 9 --max_depth 15 --num_workers 30

Notes:
    1.  use --no_paranthesis to make stored data shorter as the listops tokenizer in the LRA pipeline anyways removes paranthesis and hence using this flag will NOT affect the input to your model.
    2.  values corresponding to all sub-nodes of the tree are also saved and can optionally be used as extra supervision.
"""


import os, sys, random, jsonlines, logging, argparse, json, csv, psutil
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from string import digits
import numpy as np

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ops  - shouldn't contain whitespace
MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
FIRST = '[FIRST'
LAST = '[LAST'
SUM_MOD = '[SM'     # sum mod 10
END = ']'
NO_OP = 'NONE'

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

NON_LEAF_P = 0.25  # pr of creating a non-leaf node


def apply_op(op, l):
    if op == MIN:
        val = min(l)
    elif op == MAX:
        val = max(l)
    elif op == FIRST:
        val = l[0]
    elif op == LAST:
        val = l[-1]
    elif op == MED:
        val = np.median(l)
    elif op == SUM_MOD:
        val = sum(l) % 10
    else:
        assert False
    return int(val)


def generate_tree(depth, max_depth, max_args):
    """ Generate tree-like equations.
        Args:
            depth: current depth of the node, int.
            max_depth: maximum depth of the tree, int.
            max_args: maximum number of arguments per operator, int.
        Returns:
            (root node of a tree structure, 
            vals of all nodes in post order, 
            lengths at all nodes in post order, 
            heights of all nodes in post order, 
            ops at all nodes in post order)
    """
    if depth < max_depth:
        r = random.random()
    else:
        r = 1                  # leaf

    if r > NON_LEAF_P:         # leaf
        value = random.choice(VALUES)
        return value, [value], [1], [0], [NO_OP]
    else:
        length = 0
        height = 0              
        num_args = random.randint(2, max_args)
        args, sub_ts, sub_vals, sub_lengths, sub_heights, sub_ops = [], [], [], [], [], []
        for _ in range(num_args):
            sub_t, sub_vs, sub_ls, sub_hs, sub_os = generate_tree(depth + 1, max_depth, max_args)
            sub_ts.append(sub_t)
            
            sub_vals.extend(sub_vs)
            args.append(sub_vs[-1])
            
            sub_lengths.extend(sub_ls)
            length += sub_ls[-1]
            
            sub_heights.extend(sub_hs)
            height = max(height, sub_hs[-1])
            
            sub_ops.extend(sub_os)
            
        op = random.choice(OPERATORS)
        t = (op, sub_ts[0])
        for sub_t in sub_ts[1:]:
            t = (t, sub_t)
        t = (t, END)                 # (((..(op, sub_t_0),..), sub_t_num_args-1), END)
        
        val = apply_op(op, args)
        length += 2                  # account for '[op'  ,  ']'
        height += 1                  
        
        return t, [NO_OP] + sub_vals + [val], [NO_OP] + sub_lengths + [length], [NO_OP] + sub_heights + [height], [NO_OP] + sub_ops + [op]

    
def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    
    sl, sr = to_string(t[0], parens), to_string(t[1], parens)
    if parens:
        return f'( {sl} {sr} )'
    return f'{sl} {sr}'


def write_to_file(data, path):
    """Write to file output."""
    logger.info(type(data))
    logger.info(f'Writing {len(data)} samples to {path}.tsv')
    with open(path + '.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Source', 'Target', 'SubTreeEvals', 'SubTreeLengths', 'SubTreeHeights', 'SubTreeOps', 'SubTreeInds', 'Id'])
        writer.writerows(data)


def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    
def generate_data(worker_id, args):
    set_seed(args.seed + worker_id)
    
    process = psutil.Process(os.getpid())
    
    train, data, templates = [], set(), set()
    num_samples = args.per_worker_samples
    
    pbar = tqdm(total=num_samples)
    while len(train) < num_samples:
        tree, sub_vals, sub_lengths, sub_heights, sub_ops = generate_tree(1, args.max_depth, args.max_args)
        length = sub_lengths[-1]
        if not (args.min_length < length < args.max_length):
            continue
            
        expr = to_string(tree, not args.no_paranthesis)
        value = sub_vals[-1]
        
        # remove numbers, spaces
        # template = expr.translate({ord(k): ' ' for k in digits}).replace(' ', '')
        
        # ignore duplicates
#         if expr in data or (args.unique_templates and (template in templates)):
#             continue
        
#         data.add(expr)
#         templates.add(template)
        
        # save info of non-leaf nodes
        non_leaf_inds = [i for i, op in enumerate(sub_ops[:-1]) if op != NO_OP] + [len(sub_ops)-1]  # always include root
        
        sub_vals, sub_lengths, sub_heights, sub_ops = map(lambda l: ' '.join([str(l[ind]) for ind in non_leaf_inds]), 
                                                          (sub_vals, sub_lengths, sub_heights, sub_ops))
        
        sub_inds = ' '.join([str(ind) for ind in non_leaf_inds])
        
        train.append([expr, value, sub_vals, sub_lengths, sub_heights, sub_ops, sub_inds])
        
        approx_mem = round(args.num_workers * process.memory_info().rss * 2**-30, 1)
        
        pbar.set_description(f'[worker {worker_id} generated {len(train)} samples , mem {approx_mem}Gi]')
        pbar.update(1)

    return train 


def main():
    parser = argparse.ArgumentParser(description='For generating listops data.')    
    parser.add_argument(
        '--task', default='basic', type=str,
        help='Name of task to create.')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='random seed.')
    parser.add_argument(
        '--num_train_samples', default=96000, type=int,
        help=('Number of train samples.'))
    parser.add_argument(
        '--num_valid_samples', default=2000, type=int,
        help=('Number of test samples.'))
    parser.add_argument(
        '--num_test_samples', default=2000, type=int,
        help=('Number of test samples.'))
    parser.add_argument(
        '--max_depth', default=10, type=int,
        help=('maximum tree depth of training sequences.'))
    parser.add_argument(
        '--max_args', default=10, type=int,
        help=('maximum number of arguments per operator in training sequences.'))
    parser.add_argument(
        '--max_length', default=2000, type=int,
        help=('maximum length per sequence in training sequences.'))
    parser.add_argument(
        '--min_length', default=500, type=int,
        help=('minimum length per sequence in training sequences.'))
    parser.add_argument(
        '--no_paranthesis', action='store_true', 
        help=('exclude parans from expressions, makes them much shorter.'))
    parser.add_argument(
        '--unique_templates', action='store_true', 
        help=('consider two expressions with same tree structure duplicates.'))
    parser.add_argument(
        '--num_workers', default=1, type=int,
        help='number of cores.')
    parser.add_argument(
        '--per_worker_samples', default=-1, type=int,
        help='number of samples to generate per worker')
    parser.add_argument(
        '--output_dir', required=True, type=str,
        help='Directory to output files.')
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    num_samples = args.num_train_samples + args.num_test_samples + args.num_valid_samples
    if args.per_worker_samples <= 0:
        args.per_worker_samples = np.ceil(1.1 * num_samples / args.num_workers)
    
    with Pool(args.num_workers) as p:
        result = list(p.map(partial(generate_data, args=args), range(args.num_workers)))
        
    print('Workers done.')
    _train = []
    for r in result:
        _train += r
        del r
    
    logger.info(f'Generated {len(_train)}. \n Removing duplicates...')
    train, data, templates = [], set(), set()
    for sample in _train:
        [expr, value, sub_vals, sub_lengths, sub_heights, sub_ops, sub_inds] = sample
        
        # remove numbers, spaces
        template = expr.translate({ord(k): ' ' for k in digits}).replace(' ', '')
        
        # ignore duplicates
        if expr in data or (args.unique_templates and (template in templates)):
            continue
        
        data.add(expr)
        templates.add(template)
        train.append(sample + [len(train)])
    
    if len(train) < num_samples:
        logger.error(f"""Generated only {len(train)} instead of {num_samples} samples due to duplicity. 
                         Please explicitly provide a large value of --per_worker_samples""")
        exit()
    
    # logger.info('sorting samples by length before splitting to ensure val/test have longer samples')
    # train.sort(key=lambda x: len(x[0]))
     
    train = train[:num_samples]
    val = train[args.num_train_samples:]
    test = val[args.num_valid_samples:]
    val = val[:args.num_valid_samples]
    train = train[:args.num_train_samples]
    
    logger.info('Dataset size: %d/%d/%d' % (len(train), len(val), len(test)))
    
    os.makedirs(args.output_dir, exist_ok=True)
            
    for split_data, split_name in zip((train, val, test), ('train', 'val', 'test')):
        write_to_file(split_data, args.output_dir + f'/{args.task}_{split_name}')
         
    logger.info('Done.')
    
    for col, entry in zip(
        ['Source', 'Template', 'Target', 'SubTreeEvals', 'SubTreeLengths', 'SubTreeHeights', 'SubTreeOps', 'SubTreeInds'],
        [expr, template, value, sub_vals, sub_lengths, sub_heights, sub_ops, sub_inds]
    ):
        print(col, entry)


if __name__ == "__main__":
    main()
