# LungLens TB: AI-powered Tuberculosis Detection Tool
LungLens TB is an innovative project designed to transform tuberculosis diagnosis in low to middle-income countries (LMICs) by leveraging advanced AI technologies. This tool was developed for Philly CodeFest 2024, where it earned the first prize. It aims to alleviate the burden on healthcare systems by providing quick and accurate tuberculosis detection.

There are two main aspects in this project and they are: Imaging Tool and Reporting Tool

## Imaging Tool
The Imaging Tool provides an interface for users to upload X-ray images. Upon submission, our trained AI model analyzes the image to determine the presence of tuberculosis. The results include a heatmap overlay on the X-ray image, highlighting the regions the model focuses on, enhancing interpretability and trust in AI assessments.

<img width="1382" alt="Screenshot 2024-04-23 at 12 27 29 PM" src="https://github.com/isamu-isozaki/LungLens-TB/assets/20830075/aed84622-4154-458b-85e5-32d111a1530e">
<img width="1393" alt="Screenshot 2024-04-23 at 12 27 03 PM" src="https://github.com/isamu-isozaki/LungLens-TB/assets/20830075/160e1a06-05ac-458d-a757-e8246eb1af26">


## Reporting Tool
The Reporting Tool allows users to input radiology reports, which are then translated into language understandable by laypeople, targeted at a 6th-grade English reading level. Additionally, this simplified text can be translated into various languages, focusing on those prevalent in LMICs, making medical information more accessible globally.
<img width="1382" alt="Screenshot 2024-04-23 at 12 27 33 PM" src="https://github.com/isamu-isozaki/LungLens-TB/assets/20830075/53e508af-f70e-4cd1-8d7a-e2759df04c12">


<img width="1382" alt="Screenshot 2024-04-23 at 12 27 39 PM" src="https://github.com/isamu-isozaki/LungLens-TB/assets/20830075/efd4d064-56d4-4b71-ad02-7501cc7c7cdc">

<img width="1382" alt="Screenshot 2024-04-23 at 12 27 18 PM" src="https://github.com/isamu-isozaki/LungLens-TB/assets/20830075/b88d2502-63e7-4301-bb10-d97bc6978837">


## Training Process
To start pretraining, assuming you have pytorch, do
```
pip install accelerate timm diffusers pillow wandb transformers datasets
accelerate launch pretraining.py
```
<img width="1177" alt="Screenshot 2024-04-23 at 12 50 04 PM" src="https://github.com/isamu-isozaki/LungLens-TB/assets/20830075/a06963db-876e-474a-87db-49ed879c0ff8">


### Deployment

For deployment do

```
pip install python-dotenv flask langchain-openai grad-cam
python model_deployment.py
```

