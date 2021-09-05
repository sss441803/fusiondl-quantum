#!/bin/sh

ANCILLAS=1

for HIDDEN in 8
do
    for LAYERS in 8
    do
        for KERNEL_SIZE in 8
        do
            echo 0h${HIDDEN}l${LAYERS}k${KERNEL_SIZE}
            sbatch --job-name=0h${HIDDEN}l${LAYERS}k${KERNEL_SIZE} --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=10G --gres=gpu:2 --time=30:00:00 --mail-type=end --mail-type=fail --mail-user=hl8967@princeton.edu --wrap="python -u quantum_learn.py --subsampling 10 --tcn_type 'd' --tcn_hidden ${HIDDEN} --tcn_layers ${LAYERS} --kernel_temporal ${KERNEL_SIZE} --ancillas ${ANCILLAS} > outputs/DTCN_h${HIDDEN}l${LAYERS}k${KERNEL_SIZE}a${ANCILLAS}.out"
            sleep 1
        done
    done
done