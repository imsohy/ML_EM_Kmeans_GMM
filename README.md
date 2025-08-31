# EM Algorithm Using K-Means for GMM

## 프로젝트 개요

본 과제의 목표는 Iris 데이터셋(150개 샘플 * 4개 feature) 에 대해 Expectation-Maximization (EM) 알고리즘을 통해 train된 Gaussian Mixture Model (GMM) 을 사용해 클러스터링 알고리즘을 구현하고, 

파라미터를 학습한 뒤, 비지도 클러스터링결과를 K-means와 비교 분석하는 것이다. 초기화를 위해 랜덤 값을 대입하는 대신, K-Means 클러스터링 파라미터를 초기값으로 지정하였다.

GMM은 각각의 클러스터가 다변량 Gaussian 분포를 따를 것으로 가정하고, EM은 잠재 변수가 존재하는 Maximum likelihood estimation 과정에서 직접적으로 최적화하기 어려운 우도 함수를 반복적으로 최적화하기 위해 사용된다.

본 과제의 목적은 EM 알고리즘의 수식을 이해하고, 응용 과정까지 정리된 sudo-code를 보고 실제 코드 상에서 구현해볼 수 있도록 하는 것이다.

## 의의

python을 사용하여, Expectation-Minimization 알고리즘을 통해 train된 가우시안 믹스쳐 모델을 사용해 클러스터링 알고리즘을 구현하고, 

파라미터를 학습한 후 비지도 클러스터링 결과를 K-means와 비교 분석하였다. 계산 과정에서 수치적으로 값이 너무 커지거나 작아지지 않게 하기 위하여 수치해석적인 방법을 적용하였다.

## dataset
https://scikit-learn.org/1.4/auto_examples/datasets/plot_iris_dataset.html

## result

[<img width="1681" height="560" alt="image" src="https://github.com/user-attachments/assets/e6e27f6b-2e64-4e6e-81ae-19dba5cd8612" />](https://github.com/imsohy/ML_EM_Kmeans_GMM/blob/28cf7814ba221da47370a94f2edc597932096895/GMM/combined.png)

# accuracy

<img width="635" height="120" alt="image" src="https://github.com/user-attachments/assets/1f38919d-5fe3-472a-ad91-41752fe69dd9" />
