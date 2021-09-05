#!/bin/sh

for HIDDEN in 2
do
    for LAYERS in 2 3 4
    do
        for KERNEL_SIZE in 4
        do
            echo nhh${HIDDEN}l${LAYERS}k${KERNEL_SIZE}
            sbatch --job-name=nhh${HIDDEN}l${LAYERS}k${KERNEL_SIZE} --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=5G --gres=gpu:1 --time=10:00:00 --mail-type=end --mail-type=fail --mail-user=hl8967@princeton.edu --wrap="python -u quantum_learn.py --subsampling 10 --tcn_type 'q' --tcn_hidden ${HIDDEN} --tcn_layers ${LAYERS} --kernel_temporal ${KERNEL_SIZE} > outputs/no_higher_encoding_QTCN_h${HIDDEN}l${LAYERS}k${KERNEL_SIZE}.out"
            sleep 1
        done
    done
done