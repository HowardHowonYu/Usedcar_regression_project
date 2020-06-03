# Craiglist 데이터를 활용한 미국 중고차 가격 예측
- 유호원, 조대선, 홍성현, 배준영

## Goals
- "Craiglist(미국판 중고나라)"의 약 51만건의 미국 중고차 정보를 활용한 **가격 예측**

## Technical Skills
- Python, Scikit-learn, pandas, numpy

## Workflow

<img src="/howard/regression_project/regression_img0.png" width="1350px">


## Issue
- 데이터 특성상 허위 및 광고성 매물로 인한 주행거리, 연식등의 이상치들이 다수 존재.
- 따라서, **이상치 탐색 및 제거**가 중요.

## Issue solving
- **vin(차대 번호)** 를 활용한 이상치 탐색 작업 진행
- https://www.vinaudit.com/ 에서 제공하는 api를 이용하여 허위 매물 탐색 및 이력 조회
- 위 api는 미국 정부기관에서 관리하는 데이터베이스를 기반으로 제작, 신뢰도가 높음

## Result
- R-square 약 0.88 달성

<img src="/howard/regression_project/3.png" width="1350px">


- **가설 검증**
  - 가설 1 : 미국의 중고차도 한국과 마찬가지로 약 5만km를 기준으로 가격이 급격히 떨어질 것이다.

  <img src="/howard/regression_project/2.png" width="1350px">

  - 데이터에서 가장 많은 매물인 2012년식 포드 F-150 FX4 차량운 주행거리 3만 마일 (약 4만8천km) 지점에서 가격이 급격히 떨어진다.

  - 가설 2: 미국은 각 주별로 가격의 차이가 있을 것이다.

  <img src="/howard/regression_project/1.png" width="1350px">
  
  - 포드 F-150 FX4 차량의 워싱턴 주와 코네티컷 주의 차량 가격의 차는 약 1만불 정도 이다.
