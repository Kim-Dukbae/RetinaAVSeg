# loss function review

research paper: https://discovery.ucl.ac.uk/id/eprint/10188133/1/CF-Loss-accepted.pdf <br>
git hub : https://github.com/rmaphoh/feature-loss/blob/main/scripts/loss.py <br>
[KO]  
 이 코드는 Yukun Zhou 외 연구진의 "CF-Loss: Clinically-relevant feature optimised loss function..." (MedIA, 2024) 논문 구현을 기반으로 수정되었습니다.   <br>
원본 코드는 MIT License 하에 배포되었으며, 본 수정본은 실험 및 재현 목적에 맞게 일부 구조와 변수명을 개선한 것입니다.   <br>
원저작자: Yukun Zhou (2023)   <br>
수정자: 형준 김 (Hyungjun Kim, 2024) <br> 

[EN]  
 This code is a modified implementation based on the original work by Yukun Zhou et al.,   <br>
published in the paper *“CF-Loss: Clinically-relevant feature optimised loss function...” (MedIA, 2024)*.   <br>
The original code was released under the MIT License.   <br>
This version includes structural refactoring and variable renaming for experimental and clarity purposes.   <br>
Original author: Yukun Zhou (2023)  <br>
Modified by: Hyungjun Kim (2024) <br>

* 원 논문에서 CF-Loss는 4-class 분할 (background, artery, vein, uncertain/overlap)을 기반으로 하지만,  <br>
  본 프로젝트에서는 혈관의 lumen 관측 특성상 overlap class를 vein으로 통합하고, 3-class segmentation 구조로 구현하였습니다.
