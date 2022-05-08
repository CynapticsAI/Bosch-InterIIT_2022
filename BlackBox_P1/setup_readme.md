# Black Box Model Extraction on SwinT
## Team 17

### A. Training
#### Setup
- Download the entire folder on your local drive.
- Navigate to the folder directory, and setup the virtual environment as:
```
>python -m venv v1
>source v1/bin/activate
```
- Install necessary dependencies
```
>pip install -U torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchtext==0.9.0 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
>pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html
>pip install -r VST/requirements.txt
```
Note that the above versions have been chosen corresponding to the CUDA version on our device (=CUDA 11.2).

- Run `train.py` script. The checkpoints will be stored in the folder `checkpoint` in the base directory


Training was done on a Linux server equipped with NVIDIA Tesla V100 32GB.  
