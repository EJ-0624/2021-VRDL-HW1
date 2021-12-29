# 2021-VRDL-HW1


## Environment

* Ubuntu 18.04.5 LTS
### lsb_release -a   
### cat /etc/lsb-release

## Requirements

* python 3.6
* PyTorch 1.7.1
* cuda 10.1

To install requirements:

```setup
pip install -r requirements.txt
```
    
>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> 
```

## Evaluation

To evaluate my model, run:

```eval
python inference.py --model-file Efficientnet_b4_weights.pth 
```

## Weight for Training Model

You can download the file here:

- [The file of weight](https://drive.google.com/file/d/1gYVfYmBDfudLZ1zQC4fEsbXMn006kosy/view?usp=sharing)

## Results

Our model achieves the following performance on :

### [Bird image Classification on EfficientNet-b4]

| 5-fold CV  |  Accuracy  | 
| -----------|----------- | 
|fold 1      |   72.17%   |  
|fold 2      |   74.66%   | 
|fold 3      |   70.83%   |   
|fold 4      |   73.33%   |  
|fold 5      |   72.33%   | 

