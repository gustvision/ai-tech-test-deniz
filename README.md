# Powder AI Tech Test
Candidate Name: Deniz Engin


# Installation
To create a new conda environment:

```
conda env create -f environment.yml
```
or

```
conda create -n powder python=3.9
conda activate powder
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers
conda install pandas tqdm
conda install mypy pytest
```

# Project description

This project is design for audio classification task to detect whales.

# Usage

### Create val split

Create a val split from train split

```
python src/create_val_split.py
```

### Train Model

For a quick use `--debug` argument to load small portion of the data.

```
python src/train.py
```

### Evaluate Model
TODO: Write eval code
```
python src/eval.py
```

### Model Inference
TODO: Write inference code 
```
python src/infer.py
```

# TODO

* Create val split by preserving class ratio from train split
* Write eval metrics - auc and roc (data class distribution imbalance)
* Write inference code
* Model improvements:
  * Different audio classification models can be tested/used to get better performance
  * Maybe instead of frozen CLAP backbone, some layers can be trained (or full backbone trained according to obtaining more data)
  * Add learning rate scheduler
  * Add early stopping
  * Hyperparameter tuning
  * Investigating class imbalance problem (undersample, class based loss weight, oversampling by augmentation)
* Add monitoring (TensorBoard or Weights&Biases)
* Add unit tests
* Use type-hinting
* Use typer
* Improve comments and documentation
* Add AMP
* Add DDP




