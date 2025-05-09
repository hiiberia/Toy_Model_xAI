<div align="center">
<h1> xAI for Wildfire Prediction with Satellite Imagery </h1>
<img src="https://github.com/user-attachments/assets/fff73623-d756-438d-97b5-449370a82e21" alt="image" width="200"/>
</div>


This project presents a toy example of an explainable AI (xAI) model for wildfire prediction based on satellite imagery. The goal is to showcase how post-hoc interpretability methods can be used to enhance trust and understanding in computer vision models.

## Dataset

- **Source**: [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
- **Contents**: Satellite images from wildfire-affected and unaffected areas in Canada, particularly from southern Quebec. Images are categorized into "fire" and "no fire" classes.

## Overview

Currently, the project includes two demonstration pipelines based on post-hoc explainability methods. This README focuses on the first:

- **Model**: A simple Convolutional Neural Network (CNN) trained on wildfire satellite imagery.
- **Explainability Method**: Class Activation Mapping (CAM) is used to visualize which parts of the image most influence the model's classification decision.
- **Goal**: To align with the principles of explainable AI by revealing the model's internal reasoning process in a transparent and intuitive way.

## Set Up

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wildfire-xai-cam.git
cd wildfire-xai-cam
pip install -r conda_requirements.txt
```

## Test visualisation 

To see inference on test examples use this [notebook](./notebooks/test_visualization.ipynb).

## Launch Gradio

To launch Gradio execute this [script](./scripts/launch_gradio.py).


## Model training 

To train a model use this [script](./scripts/train.py).

## Dataset source

You can find the used dataset on [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data).

## 
