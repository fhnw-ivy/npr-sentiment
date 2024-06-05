#!/bin/bash
python /teamspace/studios/this_studio/npr-sentiment/src/model_pipeline.py eval 
python /teamspace/studios/this_studio/npr-sentiment/src/model_pipeline.py  finetune --nested-splits --batch-size 105 --num-epochs 25
python /teamspace/studios/this_studio/npr-sentiment/src/model_pipeline.py  transfer --nested-splits --batch-size 105 --num-epochs 25
python /teamspace/studios/this_studio/npr-sentiment/src/model_pipeline.py  finetune --nested-splits --weak-label-path "/teamspace/studios/this_studio/npr-sentiment/data/weak_labelled/log_reg_weaklabels.parquet" --batch-size 105 --num-epochs 25