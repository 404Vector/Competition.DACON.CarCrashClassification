# Competition.DACON.CarCrashClassification

- [Competition Link](https://dacon.io/competitions/official/236064/overview/description)
- [WandB Link](https://wandb.ai/404vector/Competition.DACON.CarCrashClassification?workspace=user-404vector)

## 개요
안녕하세요, 여러분! 😀 제1회 코스포 x 데이콘 자동차 충돌 분석 AI경진대회(채용 연계형)에 오신 것을 환영합니다.



최근 블랙박스를 활용해 도로 위에서의 사고를 예방하려는 연구가 다양하게 진행되고 있습니다.

이에 따라 근래 블랙박스는 위험 상황을 스스로 판단하고, 실시간으로 대처하는 인공지능 기능이 추가되는 방향으로 발전하고 있는데요. 

교통사고의 발생과 규모를 큰 폭으로 감소시키고, 향상된 교통 안전 기술을 개발하는 데 유용하게 활용될 수 있을 것으로 기대됩니다!



이번 월간 데이콘에서는 이러한 기술 개발의 일환으로, 블랙박스 영상으로부터 자동차의 충돌 상황을 분석하는 AI 모델을 만들어야 합니다.

컴퓨터 비전 분야의 AI 기술을 적극적으로 활용하여 문제를 해결해 보세요!


## 주제
블랙박스 영상의 자동차 충돌 분석 AI 알고리즘 개발



## 데이터 및 목표
10프레임, 5초 분량의 블랙박스 영상(mp4)들을 입력으로 받아 자동차 충돌 상황을 분석하여,

이를 13가지 상황(Class)으로 분류하는 AI 모델을 개발해야 합니다.

## 평가지표
- Macro F1 Score
- Public score : 전체 테스트 데이터 중 30%
- Private score : 전체 테스트 데이터 중 나머지 70%
