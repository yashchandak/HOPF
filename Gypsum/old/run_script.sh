
#! /bin/sh
#! /bin/bash

python run_new.py simple x - 1 1 3 &      # Node
sleep 2s
python run_new.py simple - h 0 1 3 &  # Neighbor
sleep 2s
python run_new.py simple x h 0 1 2 &   # Simple
sleep 2s
python run_new.py kipf h h 1 1 2 &       # Kipf GCN
sleep 2s
python run_new.py simple x - 0 10 2   # ICA
sleep 2s

python run_new.py maxpool h h 0 1 2 &   # Maxpool
sleep 2s
python run_new.py mul_attention x,h x,h 0 1 3   # Mul attention