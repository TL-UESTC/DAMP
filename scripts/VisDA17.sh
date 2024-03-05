#!/bin/bash

cd ..

# custom config
DATA=/xxxx/xxxxx/xxxxxx/ # you may change your path to dataset here
TRAINER=DAMP

DATASET=visda17 # name of the dataset
CFG=damp  # config file
TAU=0.5 # pseudo label threshold
U=2.0 # coefficient for loss_u
SEED=1

NAME=sr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains synthetic --target-domains real --seed ${SEED} TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}