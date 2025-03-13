# **Project**
**망막의 시신경으로부터 정맥과 동맥 구분하는 Semactic Segmentation model 만들기**

## **Abstract**
Semantic Segmentation은 Deep Learning의 핵심 분야로 이미지에서 pixel 단위로 class를 분류하는 기법입니다.  
U-Net(2015), SegNet(2016), DeepLab(2015 ~ 2018), Duck Net(2023)와 같이 빠르게 발전하고 있습니다. 그 중 Unet은 대표적
이고 많은 논문에서 비교 모델로 사용되는 모델 중 하나입니다. Segmentation 분야는 Encoder와 Decoder로 구성되어 있
으며, 각 모델들의 공통점은 pooling 과정을 거친 후 convolution의 필터 수를 2배씩 증가됩니다. 많은 수의 convolution의
필터는 보다 더 많은 정보를 담을 수 있지만, 저희 연구 결과에 따르면 256, 512, 1024와 같은 많은 수의 feature map이
필요하지 않을 수 있습니다. 이번 연구에서 Unet에 SE-block을 결합했을 때, 256~1024의 feature map을 가진 layer에서 절
반에 해당하는 Neuron을 Dead Neuron을 반환했으며 이로 인해 기존 Unet의 정확도보다 향상했음을 확인했습니다. 이를
통해 기존의 Deep Learning에서 고정적으로 증가되는 필터 수가 불필요하거나 중복된 정보를 포함할 수 있다는 가정을 가
지고 실험을 진행했습니다. 저희는 SE-block을 기존 Convolution과 곱하지 않고 하위 Index를 재학습하는 RA(Re-Activation) block을 제시합니다. 적은 수의 feature map은 Overfitting과 Underfitting으로 이어질 수 있다는 단점이 존재합
니다. 하지만 저희 모델을 ISBI 2012 dataset에 Pre-training을 사용했을 때 Dice가 0.9 이상의 값을 얻었습니다.

**Keywords: Semantic Segmentation, Data Augmentation, SE-Net, Retinal vessel, artery, vein**

## **Conclusion**
<img src="result/test.png" />
