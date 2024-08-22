# Robust Deep Object Tracking against Adversarial Attacks

:herb: Robust Deep Object Tracking against Adversarial Attacks

International Journal of Computer Vision (IJCV) 

Shuai Jia, Chao Ma, Yibing Song, Xiaokang Yang and Ming-Hsuan Yang

This work is based on 
- [Robust Tracking against Adversarial Attacks](https://arxiv.org/pdf/2007.09919.pdf) in ECCV2020 [[Project]](https://github.com/VISION-SJTU/RTAA)
- [IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking](https://arxiv.org/pdf/2103.14938.pdf) in CVPR2021 [[Project]](https://github.com/VISION-SJTU/IoUattack)

## Introduction
<img src="https://github.com/joshuajss/rtaav2/blob/main/img/pipeline.png" width='700'/><br/>

Deep neural networks (DNNs) are vulnerable to adversarial attacks. 
- We study the robustness of the state-of-the-art deep trackers against adversarial attacks under both white-box and black-box settings. 
- We propose a defense method to subtract perturbations from the input frame, which eliminates performance drops caused by adversarial attacks. 
- We craft universal adversarial perturbations to directly inject them into every frame of video sequences, leading to a higher attack speed.
- We choose five representative trackers, [SiamRPN++](https://github.com/STVIR/pysot), [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), [RT-MDNet](https://github.com/IlchaeJung/RT-MDNet), [DiMP](https://github.com/visionml/pytracking) and [TransT](https://github.com/chenxin-dlut/TransT).

## Demo
<img src="https://github.com/joshuajss/rtaav2/blob/main/img/human9_attack.gif" width='300'/>   <img src="https://github.com/joshuajss/rtaav2/blob/main/img/human9_defense.gif" width='300'/><br/>
<img src="https://github.com/joshuajss/rtaav2/blob/main/img/legend.png" width='600'/><br/>

:herb: **More demos are available at [[Video]](https://drive.google.com/file/d/1REyTfmRvMdE9DnFHHIPBUHD0qVP3RmNO/view?usp=sharing)
 .**

## Prerequisites 

 The environment follows the tracker you intend to attack：
 
 - The specific setting and pretrained model for **SiamPRN++** can refer to [[Code_SiamRPN++]](https://github.com/STVIR/pysot).
 - The specific setting and pretrained model for **SiamCAR** can refer to [[Code_SiamCAR]](https://github.com/ohhhyeahhh/SiamCAR).
 - The specific setting and pretrained model for **RT-MDNet** can refer to [[Code_RT-MDNet]](https://github.com/IlchaeJung/RT-MDNet).
 - The specific setting and pretrained model for **DiMP** can refer to [[Code_DiMP]](https://github.com/visionml/pytracking).
 - The specific setting and pretrained model for **TransT** can refer to [[Code_TransT]](https://github.com/chenxin-dlut/TransT).
 
## Experiments
 - #### Results on the OTB2015 dataset
 <img src="https://github.com/joshuajss/rtaav2/blob/main/img/results_otb.png" width='1000'/><br/>
  - #### Results on the UAV123 dataset
 <img src="https://github.com/joshuajss/rtaav2/blob/main/img/results_uav.png" width='1000'/><br/>
  - #### Results on the LaSOT dataset
 <img src="https://github.com/joshuajss/rtaav2/blob/main/img/results_lasot.png" width='1000'/><br/>

 
:herb: **All raw results are available.**  [[Google_drive]](https://drive.google.com/file/d/1RtKwxSi6iKhsApeG3-xqP6hlvOar-Uq4/view?usp=sharing)

## Quick Start

:herb: **The code of adversarial attack and defense on SiamRPN++ is released.**

- Please follow [SiamRPN++](https://github.com/STVIR/pysot) to finish the experimental setting, including dataset, model, environment, etc.
- First, put ```test_attack.py``` and ```test_defense.py``` into ```tools``` folder.
- Second, replace the original ```siamrpn_tracker.py``` in ```pysot/tracker``` with our new ```siamrpn_tracker.py```.
- Note that the new ```siamrpn_tracker.py``` in this project consists of all original codes of SiamRPN++ and our new attack and defense code.


Test the original performance on OTB100 dataset, please using the follwing command.
```bash
cd experiments/siamrpn_r50_l234_dwxcorr_otb
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset OTB100 	\ # dataset name
	--config config.yaml	  # config file
```

Test the adversarial attack performance on OTB100 dataset, please using the follwing command.
```bash
cd experiments/siamrpn_r50_l234_dwxcorr_otb
python -u ../../tools/test_attack.py 	\
	--snapshot model.pth     	\ # model path
	--dataset OTB100         	\ # dataset name
	--config config.yaml	          # config file
```


Test the adversarial defense performance on OTB100 dataset, please using the follwing command.
```bash
cd experiments/siamrpn_r50_l234_dwxcorr_otb
python -u ../../tools/test_defense.py 	\
	--snapshot model.pth     	\ # model path
	--dataset OTB100         	\ # dataset name
	--config config.yaml	          # config file
```

- The original/attack/defense results will be saved in the current directory(results/dataset/model_name/).
- ```--vis``` can be used to visualize the tracking results during attack and defense.


## Acknowledgement 
We sincerely thanks the authors of [SiamRPN++](https://github.com/STVIR/pysot), [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), [RT-MDNet](https://github.com/IlchaeJung/RT-MDNet), [DiMP](https://github.com/visionml/pytracking) and [TransT](https://github.com/chenxin-dlut/TransT), who provide the baseline trackers for our attack and defense.
## License
Licensed under an MIT license.

