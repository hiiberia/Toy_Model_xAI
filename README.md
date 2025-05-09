# Toy POC

## Conda Set Up

```bash
# Conda env setup
conda create env --name toy-model python=3.12
echo 'conda activate toy-model' >> ~/.bashrc
source ~/.bashrc
conda install --file conda_requirements.txt

# Create the kernel for the notebook
python -m ipykernel install --user --name toy-model --display-name "Python 3.12 (toy-model)"
```

## Test visualization 

To see inference on test examples use this [notebook](./notebooks/test_visualization.ipynb).

## Launch Gradio

To launch Gradio execute this [script](./scripts/launch_gradio.py).


## Model training 

To train a model use this [script](./scripts/train.py).

## Dataset source

You can find the used dataset on [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data).