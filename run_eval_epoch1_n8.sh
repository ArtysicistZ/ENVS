#!/bin/bash
cd /home/kevinzyz/yincheng/arpo
.venv/bin/python3 -m verl.trainer.main \
    config=configs/sft_eval_300tasks_clean_n8.yaml \
    worker.actor.model.model_path=checkpoints/sft_86tasks/epoch_1 \
    trainer.experiment_name=sft_eval_epoch1_300tasks_n8
