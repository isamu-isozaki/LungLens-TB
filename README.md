# LungLens TB: AI-powered Tuberculosis Detection Tool
LungLens TB is an innovative project designed to transform tuberculosis diagnosis in low to middle-income countries (LMICs) by leveraging advanced AI technologies. This tool was developed for Philly CodeFest 2024, where it earned the first prize. It aims to alleviate the burden on healthcare systems by providing quick and accurate tuberculosis detection.



### Pretraining
To start pretraining, assuming you have pytorch, do
```
pip install accelerate timm diffusers pillow wandb transformers datasets
accelerate launch pretraining.py
```

### Deployment

For deployment do

```
pip install python-dotenv flask langchain-openai grad-cam
python model_deployment.py
```
