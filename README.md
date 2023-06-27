# Instalation

- Download the checkpoint for the EfficientNet [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz) and unzip it in the efficientnetv2-m folder.
- Download the Anatomical Regions files from the Endomapper dataset and the video into the Data Folder.
- Extract the data with the Extract.sh script and create the dataset structure using the Python script CreateDataset.py

# Usage
- To train a model:
  python train_t.py trained_efficient efficient
- To test a Model:
  python Test.py trained_efficient efficient
