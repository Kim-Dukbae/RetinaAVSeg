# Loss Function Review

| No. | Loss Function |
|-----|---------------|
|  1  | [CF Loss](#cf-loss) |
|  2  | [CL Dice](#cl-dice) |

---

## CF Loss

- 📄 Research Paper: [CF-Loss: Clinically-relevant feature optimised loss function (MedIA, 2024)](https://discovery.ucl.ac.uk/id/eprint/10188133/1/CF-Loss-accepted.pdf)  
- 💻 GitHub Source: [feature-loss/loss.py](https://github.com/rmaphoh/feature-loss/blob/main/scripts/loss.py)

> **Note**: The original paper defined CF-Loss for 4-class segmentation (background, artery, vein, uncertain/overlap).  
> In this project, due to the lumen-based vessel segmentation characteristics, the overlap class was merged with the vein class, implementing a 3-class segmentation approach.

**[KO]**  
&nbsp;이 코드는 Yukun Zhou 외 연구진의 *“CF-Loss: Clinically-relevant feature optimised loss function...”* (MedIA, 2024) 논문 구현을 기반으로 수정되었습니다. 원본 코드는 MIT License 하에 배포되었으며, 본 수정본은 실험 및 재현 목적에 맞게 일부 구조와 변수명을 개선한 것입니다.  
- 원저작자: Yukun Zhou (2023)  
- 수정자: 형준 박 (Hyungjun Kim, 2024)  

**[EN]**  
&nbsp;This code is a modified implementation based on the original work by Yukun Zhou et al., published in the paper *“CF-Loss: Clinically-relevant feature optimised loss function...”* (MedIA, 2024). The original code was released under the MIT License. This version includes structural refactoring and variable renaming for experimental and clarity purposes.  
- Original author: Yukun Zhou (2023)  
- Modified by: Hyungjun Park (2024)

---

## CL Dice

- 📄 Research Paper:[clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/pdf/2003.07311)
- 💻 GitHub Source: [jocpae/clDice](https://github.com/jocpae/clDice)

