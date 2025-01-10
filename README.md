# Synergistic Learning with Multi-Task DeepONet for Efficient PDE Problem Solving  

This repository contains the source code and resources associated with the manuscript:  
[*Synergistic Learning with Multi-Task DeepONet for Efficient PDE Problem Solving*](https://arxiv.org/abs/2408.02198). 

https://github.com/user-attachments/assets/768e7485-31e8-4dcb-85e7-155c9f882495

## Overview  
The repository demonstrates the implementation of Multi-Task DeepONet for solving partial differential equations (PDEs) efficiently through synergistic learning. The codebase is divided into three segments based on the problem discussed in the manuscript.

## Installation  

### Prerequisites  
- [Conda](https://docs.conda.io/en/latest/)  
- Python 3.8 or higher  

### Setting up the Environment  
1. Clone the repository:  
   ```bash
   git clone https://github.com/varunsingh88/MT-DeepONet.git  
   cd MT-DeepONet
   conda env create -n "environment name" -f environment.yaml
   conda activate "environment name"
   ````

### Downloading data
All dataset used in this study are available here
[*MT-DeepONet data*](https://drive.google.com/drive/folders/1HxjdCUGmbpzzBDo01JRdsT2Uk3AmZOPr?usp=sharing)

Download individual data for each problem and store in the corresponding 'Data' folders.

## Usage
Individual problems contained in the folders can be run independently after activating the conda environment: 
````bash
python main.py
````
Checkpoints saved for each problem can be used for evaluating the model's output.

## Reference
If you use this repository, please cite the manuscript:
````bibtex
@article{KUMAR2025107113,
title = {Synergistic learning with multi-task DeepONet for efficient PDE problem solving},
journal = {Neural Networks},
volume = {184},
pages = {107113},
year = {2025},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.107113},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024010426},
author = {Varun Kumar and Somdatta Goswami and Katiana Kontolati and Michael D. Shields and George Em Karniadakis},
}
````  



