#!/bin/sh

ANCILLAS=3

for HIDDEN in 2 4 8
do
    for LAYERS in 2 4 6
    do
        for KERNEL_SIZE in 4 8
        do
            echo h${HIDDEN}l${LAYERS}k${KERNEL_SIZE}
            sbatch --job-name=${ANCILLAS}h${HIDDEN}l${LAYERS}k${KERNEL_SIZE} --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=20G --time=20:00:00 --mail-type=end --mail-type=fail --mail-user=hl8967@princeton.edu --wrap="python -u quantum_learn.py --subsampling 10 --tcn_type 't' --tcn_hidden ${HIDDEN} --tcn_layers ${LAYERS} --kernel_temporal ${KERNEL_SIZE} > outputs/TTCN_h${HIDDEN}l${LAYERS}k${KERNEL_SIZE}.out"
            sleep 1
        done
    done
done