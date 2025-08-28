# MGCL:Memory guided contrastive learning for cancer subtype identification based on multi-omics data

# Dependencies
MGCL is built with Python 3.7 with the following packages:
* torch == 1.13.1
* snfpy == 0.2.2
* pandas == 1.3.5
* numpy == 1.21.6
* scikit-learn == 1.0.2
  
# Dataset
The dataset can be downloaded at  https://pan.baidu.com/s/1V5YlY59f4Sm78GcT7Io4_Q. The directory should look like


<pre> ``` Data/  ├── BLCA/ │ ├── BRCA/ │ ├── KIRC/ │├── LUAD/ │├── PAAD/ │├── SKCM/ │├── STAD/ │└── UCEC/ ``` </pre>

# Usage
We provide the scripts for running MGCL.  

```
python train.py



