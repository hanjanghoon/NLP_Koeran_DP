#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/analyze.py --parser stackptr --beam 10 --ordered --gpu \
 --punctuation '.' '``' "''" ':' ',' \
 --display \
 --test "data/sejong/test.conllx" \
 --model_path "models/stack_ptr/92.05" --model_name 'network.pt'
