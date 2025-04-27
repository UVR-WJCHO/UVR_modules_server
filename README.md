## Installation
### Requirements
- Python-3.8
- CUDA 11.7
- requirements.txt

## Setup - HandTracker

- Download pretrained model and Mano data (updated : 23/10/16)

```
https://www.dropbox.com/scl/fi/mgtnhommqvrvm2exjbxjx/SAR_AGCN4_cross_wBGaug_extraTrue_resnet34_s0_Epochs50.zip?rlkey=pgxx00s6efc3jswutzessyafl&st=297m7cxy&dl=0
```
```
https://www.dropbox.com/scl/fi/60hzlehmd74e2c3xo2pxz/mano.zip?rlkey=mrxkbn9yl06zmop6ml6n1ofsy&st=7r8izg3h&dl=0
```

- Locate the file at 
```
WISEUIServer/handtracker/checkpoint/[model_name]/checkpoint.pth
WISEUIServer/handtracker/mano_data/mano
```

- Create folder 'calibration'


- Check the path of pretrained model in `WISEUIServer/handtracker/config.py`, `checkpoint` parameter

- run 
```
activate [virtualenv]
cd ./WISEUIServer
python main.py
```