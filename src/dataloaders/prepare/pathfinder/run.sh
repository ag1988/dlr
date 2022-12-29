#!/bin/bash

# rm -rf data/

# 128, 256, 512
for i in {0..39}
do
    python snakes2_wrapper.py $i 2500 &
done


# to kill all
# ps auxww | grep 'python snakes2' | awk '{print $2}' | xargs kill -9

