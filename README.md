# LungLens

### Pretraining
To start pretraining, assuming you have pytorch, do
```
pip install accelerate timm diffusers pillow wandb transformers datasets
accelerate launch pretraining.py
```