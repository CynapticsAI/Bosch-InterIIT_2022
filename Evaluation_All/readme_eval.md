# Evaluation of extracted models
## Team 17

#### A. Setup
- Download the Kinetics 400 and Kinetics 600 datasets (validations splits) and place them on the base directory. For each of the above two datasets, the directory structure should be as follows:
```
|-- k400_val
|   |-- air drumming
|   |   |-- 2cPLjY5AWXU.mp4
|   |   |-- 3K0Sw7rbzPU.mkv
|   |   |-- ...
|   |-- answering questions
|   |   |-- 0htNc0u9TDk.mp4
|   |   |-- 2wA2ZT_1gdc.mp4
|   |   |-- ...
|   |   ...
```
```
|-- k600_val
|   |-- air drumming
|   |   |-- 2cPLjY5AWXU.mp4
|   |   |-- 3K0Sw7rbzPU.mkv
|   |   |-- ...
|   |-- answering questions
|   |   |-- 0htNc0u9TDk.mp4
|   |   |-- 2wA2ZT_1gdc.mp4
|   |   |-- ...
|   |   ...
```
We provide the following links to download these datasets:

> Kinetics 400: [Drive Link 1](https://drive.google.com/drive/folders/1-5K4BxC0GbBe9x6Wxjzdd6orpANkKab1?usp=sharing)
> Kinetics 600: [Drive Link 2](https://drive.google.com/drive/folders/1nDsW-C09gO_eSD-6cJLow8zN9wSFAWDK?usp=sharing)

- Put the checkpoints of the trained models in the `checkpoints` folder. Our checkpoints for P1 and P2 models for both the Black and Grey Box settings can be accessed at [Checkpoints](https://drive.google.com/drive/folders/1YKL7C75yISUQPQV-8ITSDuHuvV__O8mJ?usp=sharing).  

#### B. Evaluation
Separate jupyter scripts have been prepared for evaluation of both P1 and P2 models under the Black Box and the Grey Box settings. Ensure that the paths at the beginning of each script are correct and redirect to the concerned location. The scripts are:

- **eval_BlackBox_P1.ipynb**
- **eval_BlackBox_P2.ipynb**
- **eval_GreyBox_P1.ipynb**
- **eval_GreyBox_P2.ipynb**

A single Colab Notebook combining the above four files is created with the name - `HP_BOSCH_Code_T17.ipynb`.  

We have evaluated our models using Colab Notebooks with the Colab Pro Plus subscription.    

