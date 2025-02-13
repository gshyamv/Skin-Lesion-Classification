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
snapshot_download(repo_id=model_name, local_dir="/content/attention-unet")