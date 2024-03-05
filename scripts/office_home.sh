#!/bin/bash

cd ..

# custom config
DATA=/xxxx/xxxxx/xxxxxx/ # you may change your path to dataset here
TRAINER=DAMP

DATASET=office_home # name of the dataset
CFG=damp  # config file
TAU=0.6 # pseudo label threshold
U=1.0 # coefficient for loss_u
SEED=1

NAME=ac
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=ap
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains product --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=ar
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains real_world --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=ca
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains art --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=cp
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains product --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=cr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains real_world --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=pa
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains art --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=pc
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=pr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains real_world --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=ra
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains art --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=rc
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=rp
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains product --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}