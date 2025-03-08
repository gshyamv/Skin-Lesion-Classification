# Skin-Lesion-Classification
## Download the requirements
```shell
pip install -r requirements.txt
```
## Datasets used in training of the models:
<li>HAM10000</li>
<li>DermMel</li>
<li>Med Node</li>
<li>SD260</li>
<li>Skin Cancer ISIC</li>

## Download the pretrained models:
### Attention U-Net
```python
from huggingface_hub import snapshot_download

# Define the model repo
model_name = "Sharukesh/attention-unet"

# Download the model locally
snapshot_download(repo_id=model_name, local_dir="/content/attention-unet")
```

### GAN
```python
from huggingface_hub import snapshot_download

# Define the model repo
model_name = "Sharukesh/GAN-HAM10000-class-balancing"

# Download the model locally
snapshot_download(repo_id=model_name, local_dir="/content/GAN")
```

### SMOTE
On our implementation of GAN the outputs were not well featurized, so those images could not be used in the training of the model, hence we choose to do SMOTE (Synthetic Minority Oversampling Technique).

#### How SMOTE Works:

1. **Identify Minority Class:** It targets the minority class in an imbalanced dataset.
2. **Select a Sample:** Randomly picks a sample from the minority class.
3. **Find Nearest Neighbors:** Identifies its k-nearest neighbors in the feature space (typically using Euclidean distance).
4. **Generate Synthetic Samples:** Creates new synthetic data points by interpolating between the original sample and one of its nearest neighbors.
5. **Repeat:** This process is repeated until the desired class balance is achieved.

Find our implementation of smote down [here](https://github.com/gshyamv/Skin-Lesion-Classification/tree/main/SMOTE)
