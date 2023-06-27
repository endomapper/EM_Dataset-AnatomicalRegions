#/bin/bash
python train_t.py trained_efficient efficient
python Test.py trained_efficient efficient
python train_t.py trained_mobile mobile
python Test.py trained_mobile mobile
python train_t.py trained_densenet densenet
python Test.py trained_densenet densenet
python train_t.py trained_resnet resnet
python Test.py trained_resnet resnet
