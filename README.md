# Synergistic Learning with Multi-Task DeepONet for Efficient PDE Problem Solving  

This repository contains the source code and resources associated with the manuscript:  
[*Synergistic Learning with Multi-Task DeepONet for Efficient PDE Problem Solving*](https://arxiv.org/abs/2408.02198).  

## Overview  
The repository demonstrates the implementation of Multi-Task DeepONet for solving partial differential equations (PDEs) efficiently through synergistic learning. The codebase is divided into three segments based on the problem discussed in the manuscript.

## Installation  

### Prerequisites  
- [Conda](https://docs.conda.io/en/latest/)  
- Python 3.8 or higher  

### Setting up the Environment  
1. Clone the repository:  
   ```bash
   git clone https://github.com/varunsingh88/MTL_DeepONet.git  
   cd MTL_DeepONet
   conda env create -n "environment name" -f environment.yaml
   conda activate "environment name"
   ````

### Downloading data
The dataset for experiments in this study is available here
[*Download MT-DeepONet data*](https://drive.google.com/drive/folders/1HxjdCUGmbpzzBDo01JRdsT2Uk3AmZOPr?usp=sharing)

Download individual data for each problem and store in the corresponding 'Data' folder for each problem.

## Usage
Individual problems contained in the folders can be run independently after activating the conda environment: 
````bash
python main.py
````
Checkpoints saved for each problem can be used for evaluating the model's output.

## Reference
If you use this repository, please cite the manuscript:
````bibtex
@article{kumar2024synergistic,
  title={Synergistic Learning with Multi-task DeepONet for Efficient PDE Problem Solving},
  author={Kumar, Varun and Goswami, Somdatta and Kontolati, Katiana and Shields, Michael D and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2408.02198},
  year={2024}
}
````  



