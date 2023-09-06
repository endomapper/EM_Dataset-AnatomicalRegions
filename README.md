# EM_Dataset-AnatomicalRegions

This repository contains the software for the EndoMapper dataset validation described in the section "Anatomical region validation" of the paper:

    Azagra P. et al. Endomapper dataset of complete calibrated endoscopy procedures. Scientific Data. 2023. Accepted for publication
# Instalation
- Download the EfficientNet model from the [Github](https://github.com/google/automl/tree/master/efficientnetv2) and add it into the efficient folder.
- Download the checkpoint for the EfficientNet [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz) and unzip it in the efficientnetv2-m folder.
- Download the Anatomical Regions files from the Endomapper dataset and the video into the Data Folder.
- Extract the data with the Extract.sh script and create the dataset structure using the Python script CreateDataset.py

# Usage
- To train a model:
  python train_t.py trained_efficient efficient
- To test a Model:
  python Test.py trained_efficient efficient
