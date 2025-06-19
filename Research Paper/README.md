# Related work 
## 📚 Retinal Vessel Segmentation Datasets

| Dataset     | Description                         | Labeling                 | Resolution / #Images    | Link | Difficulty |
|-------------|-------------------------------------|--------------------------|--------------------------|------|------------|
| **DRIVE**   | Diabetic retinopathy fundus images  | Vessel                   | 565×584 / 40             | [🔗 Dropbox](https://www.dropbox.com/scl/fo/2y1tt4t7l939d37w6diyx/AC2PXSmqEBJUR1n1zFIbrE4?rlkey=5lqh1o0h22l08vpyjcddfjs8x&e=1&dl=0) | ★★★☆☆ |
| **STARE**   | Pathology-included wide-angle views | Vessel       | 700×605 / 20             | [🔗 Clemson](https://cecas.clemson.edu/~ahoover/stare/probing/index.html) | ★★★★☆ |
| **CHASE_DB1** | Pediatric eye fundus dataset       | Vessel      | 999×960 / 28             | [🔗 Kingston Univ.](https://researchdata.kingston.ac.uk/96/) | ★★★★☆ |
| **LES-AV**  | Labeled A/V segmentation ground truth | Vessel / Artery / Vein          | 768×584 / 22             | [🔗 Figshare](https://figshare.com/articles/dataset/LES-AV_dataset/11857698?file=21732282) | ★★★★★ |
| **HRF**     | High-res images with pathology      | Vessel / Non-vessel      | 3504×2336 / 45           | [🔗 FAU Dataset](https://www5.cs.fau.de/research/data/fundus-images/) | ★★★★☆ |
| **IOSTAR**  | Fundus via SLO/OCT fusion           | Artery / Vein / Background | 1024×1024 / 30        | [🔗 RetinaCheck](https://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset) | ★★★★★ |
| **ROSE**    | OCTA-based capillary segmentation   | Capillaries included      | 512×512 / 117            | [🔗 NIMTE](https://imed.nimte.ac.cn/dataofrose.html) | ★★★★★ |
| **OCTA-500**| 3D→2D projected retinal vasculature | Deep capillary plexus     | Varies / 500 subjects    | [🔗 IEEE DataPort](https://ieee-dataport.org/open-access/octa-500) | ★★★★★ |
| **2PFM**    | 2-Photon microscopy (rat retina)     | Microvessels              | 1024×1024 / 115          | [🔗 Figshare](https://figshare.com/articles/dataset/2PFM_dataset_from_MaskVSC/28203014) | ★★★★★ |
| **RAVIR**   | Vascular segmentation (rat retina)   | Artery / Vein / Others    | 768×768 / 23             | [🔗 Grand Challenge](https://ravir.grand-challenge.org/data/) | ★★★★★ |

## 📊 Matrix
| 정확도  | Dice |
|-------------|-----------------|
| 연속성  | cl Dice |
| 임상적 지표  | CF-Loss |

> 임상 지식을 딥러닝이 학습하도록 유도하되, 모델 내부는 여전히 representation learning의 자유를 보장해야 한다. CF-Loss가 딥러닝 모델을 학습시키는 loss function보다는, 예측 결과를 평가하는 임상적 지표로 활용하는 것이 더 적합하다.
