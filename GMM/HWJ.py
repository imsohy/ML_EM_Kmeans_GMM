# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.image as mpimg # 이미지 불러올 때 사용

# Seed setting
#seed_num = 25208                # 재현을 위해 시드고정
seed_num = 80000
np.random.seed(seed_num)
iteration = 100                 #반복 호출횟수 변수

class EM:
    """
    expectation-maximization algorithm, EM algorithm
    The EM class is a class that implements an EM algorithm using GMM and kmeans.
    
    Within the fit function, the remaining functions should be used.
    Other functions can be added, but all tasks must be implemented with Python built-in functions and Numpy functions.
    You should annotate each function with a description of the function and parameters(Leave a comment).
    """

    def __init__(self, n_clusters, iteration):
        """
        Parameters
        ----------
        n_clusters (int): Num of clusters (num of GMM)
        iteration (int): Num of iteration 
            Termination conditions if the model does not converge
        mean (ndarray): Num of clusters x Num of features
            The mean vector that each cluster has.
        sigma (ndarray): Num of clusters x Num of features x Num of features     
            The covariance matrix that each cluster has.
        pi (ndarray): Num of labels (num of clusters)
            z(x), Prior probability that each cluster has.
        gamma (ndarray): 사후확률 행렬 (N, K)
            각 샘플이 k번째 클러스터에 속할 확률 행렬.
            부가 설명:
            젠센부등식에서 도입된 보조분포이다.
        return None.
        -------
        None.

        """
        # n_cluster, iteration 초기화
        self.n_clusters = n_clusters    
        self.iteration   = iteration
        # 파라미터들은 변수의 공간만 할당하고, 초기화는 initialization 함수에서 수행하기로 한다
        self.mean = None            # shape: (클러스터수 K, 차원 D) = (3,4) for iris dataset
        self.sigma = None           # shape: (K, D, D)
        self.pi    = None           # shape: (K,)
        self.gamma = None           # shape: (N, K)

    def initialization(self, X):
        """
        초기화 함수 (initialization)
        KMeans 결과를 바탕으로 GMM의 초기 파라미터(mean, sigma, pi)를 설정

        Parameters
        ----------
        X (ndarray): (N, D) 크기의 입력 데이터
            N은 샘플 개수, D는 특성(피처)의 개수
        self.mean (ndarray): (K, D)
            K개의 클러스터 중심(평균 벡터)
        self.sigma (ndarray): (K, D, D)
            K개의 공분산 행렬
        self.pi (ndarray): (K,)
            각 클러스터에 대한 사전 확률
        self.gamma (ndarray): (N, K)
            각 샘플이 k번째 클러스터에 속할 사후확률.
            
        return None.
        -------
        None.
        """
        N, D = X.shape
        # 1) K-Means clustering으로 초기화.
        km = KMeans(n_clusters=self.n_clusters, init="random", random_state=seed_num).fit(X)       #kmeans clustering
        labels = km.labels_                   # shape: (N,)
        centers = km.cluster_centers_         # shape: (K, D)
        # 2) Kmeans cluster centers를 초기 평균으로 사용
        self.mean = centers.copy()            # shape: (K, D)

        # 3) covariance 행렬과 mixing coefficient 초기화
        self.sigma = np.zeros((self.n_clusters, D, D))  # shape: (K, D, D)
        self.pi = np.zeros(self.n_clusters)  # shape: (K,)

        for k in range(self.n_clusters):
            Xk = X[labels == k]               # 클러스터 k 라벨의 모든 points 저장.
            Nk = Xk.shape[0]                  # 클러스터 k에 속하는 샘플의 개수
            if Nk <= 1:
                # 이 클러스터에 데이터가 없거나 1개일경우, 계산을 위해서 코배리언스를 매우 작은값으로 설정.
                self.sigma[k] = np.eye(D) * 1e-6
            else:
                # 공분산행렬 계산 + 정규화
                cov_k = np.cov(Xk, rowvar=False) + np.eye(D) * 1e-6
                # 행렬이 (D,D)가 되도록 prove함
                if cov_k.ndim == 1: 
                    cov_k = np.diag(cov_k)
                self.sigma[k] = cov_k  # 공분산 행렬 저장
            # π_k = N_k / N 계산
            self.pi[k] = Nk / N

        # 4) 감마(0) = 0으로 초기화 
        self.gamma = np.zeros((N, self.n_clusters))  # shape: (N, K)

    def multivariate_gaussian_distribution(self, X, mu, sigma):
        """
        다변량 정규분포 확률 밀도 함수 계산
        주어진 평균(mu)과 공분산(sigma)에 대해 각 샘플 X의 확률 밀도를 계산.

        Parameters
        ----------
        X (ndarray): (N, D)
            입력 샘플 데이터
        mu (ndarray): (D,)
            평균 벡터
        sigma (ndarray): (D, D)
            공분산 행렬

        return pdf (ndarray): (N,)
        -------
        각 샘플에 대한 확률 밀도값
        """
        D = X.shape[1]
        # singular matrix가 되는 걸 막기 위해, 공분산 행렬에 작은 값을 더해주기
        sigma_reg = sigma + np.eye(D) * 1e-6
        # 공분산 행렬의 역행렬과 행렬식 계산
        det_sigma = np.linalg.det(sigma_reg)
        inv_sigma = np.linalg.inv(sigma_reg)
        # 정규화 상수 계산
        norm_const = np.sqrt(((2 * np.pi) ** D) * det_sigma)
        # multivariate Gaussian PDF 계산
        diff = X - mu                               # 편차
        # -0.5 * (x - mu)^T * inv_sigma * (x - mu)
        exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
        pdf = np.exp(exponent) / norm_const         # Gaussian PDF에 값 대입
        return pdf

    def expectation(self, X):
        """
        E-step.
        현재 파라미터를 사용하여 각 데이터가 각 클러스터에 속할 확률(책임도)를 계산

        Parameters
        ----------
        X (ndarray): (N, D)
            입력 샘플 데이터
        self.gamma (ndarray): (N, K)
            각 샘플이 k번째 클러스터에 속할 책임도

        return None.
        -------
        None.
        """

        N, _ = X.shape
        # 각 클러스터 k에 대해 likelihood multivariant Gaussian PDF 계산
        pdf_matrix = np.zeros((N, self.n_clusters))
        for k in range(self.n_clusters):
            pdf_matrix[:, k] = self.multivariate_gaussian_distribution(
                X, self.mean[k], self.sigma[k]
            )
        # 사전확률 π_k를 이에 곱해 각 클러스터 k의 gamma 분자 계산
        weighted_pdfs = pdf_matrix * self.pi       # shape: (N, K)
        # 각 샘플 xn에 대해, 모든 클러스터 k에 대한 사후확률이 1이 되게 정규화.
        denom = np.sum(weighted_pdfs, axis=1, keepdims=True)  # shape: (N, 1)
        denom[denom == 0] = 1e-12  # 0 나누기 방지값
        self.gamma = weighted_pdfs / denom         # shape: (N, K)

    def maximization(self, X):
        """
        M-step
        E-step에서 계산된 gamma를 이용하여,
        평균, 공분산, 혼합 계수를 업데이트한다.

        Parameters
        ----------
        X (ndarray): (N, D)
            입력 샘플 데이터
        self.mean (ndarray): (K, D)
            업데이트된 평균 벡터
        self.sigma (ndarray): (K, D, D)
            업데이트된 공분산 행렬
        self.pi (ndarray): (K,)
            업데이트된 혼합 계수

        return None.
        -------
        None.
        """
        N, D = X.shape
        # N_k = Sigma_n gamma_{n,k}
        Nk = np.sum(self.gamma, axis=0)  # shape: (K,)

        # Update means mu_k
        self.mean = (self.gamma.T @ X) / Nk[:, np.newaxis]  # shape: (K, D)

        # Update covariances Sigma_k
        new_sigma = np.zeros((self.n_clusters, D, D))
        for k in range(self.n_clusters):
            diff = X - self.mean[k]                      # shape: (N, D)
            gamma_diag = self.gamma[:, k][:, np.newaxis] # shape: (N, 1)
            new_sigma[k] = (diff * gamma_diag).T @ diff / Nk[k]  # shape: (D, D)
            # 값이 너무 작아지는 걸 방지.
            new_sigma[k] += np.eye(D) * 1e-6
        self.sigma = new_sigma

        # 사전확률 업데이트
        self.pi = Nk / N  # shape: (K,)

    def fit(self, X):
        """
        EM 알고리즘 실행
        KMeans 기반 초기화 후, EM 알고리즘을 통해 클러스터링을 반복적으로 수행

        Parameters
        ----------
        X (ndarray): (N, D)
            입력 데이터 (레이블 없이 비지도 학습 수행.)

        return labels (ndarray): (N,)
        -------
        각 샘플에 대해 최종적으로 할당된 클러스터 번호 (0~K-1)
        """  
        N, _ = X.shape
        # 1) Initialization via K-Means
        self.initialization(X)
        prev_gamma = np.zeros_like(self.gamma)

        for i in range(self.iteration):
            #print(f"Iteration {i+1}")             # DEBUG
            # 2) E-step
            self.expectation(X)
            # 3) Gamma의 수렴 확인 (종료조건: gamma 변화 tolerance = 1e-4)
            if np.allclose(self.gamma, prev_gamma, atol=1e-4):
                break
            prev_gamma = self.gamma.copy()  # 딥 카피
            # 4) M-step
            self.maximization(X)

            #print(f"log-likelihood approx: {np.sum(np.log(np.sum(self.gamma, axis=1)))}") # DEBUG
        # 수렴 후, 각 샘플에 대해 가장 높은 확률을 갖는 클러스터를 머신러닝 결과로 할당
        return np.argmax(self.gamma, axis=1)

def plotting(df, title='pairplot', fname=None):
    """
    클러스터링 결과 시각화
    데이터프레임의 seaborn pairplot을 생성하고, df['labels']에 따라 색상을 지정하며, 
    대각선에는 연속확률변수 밀도 곡선을 표시한다.
    요구사항에 보인 대로 3개의 plot (original, EM, KMeans) 모두 표시한다.

    Parameters
    ----------
    df (pd.DataFrame): (N, D+1)
        'labels' 컬럼을 포함한 데이터프레임
    title (str): 
        plot 상단에 표시할 제목
    fname (str or None): 
        저장할 파일 이름. None이면 저장하지 않고 단순 표시

    return None.
    -------
    None.
    """
    # 1) 기본 pairplot 생성
    g = sns.pairplot(df, hue='labels', 
                     diag_kind='kde',
                     plot_kws={'alpha':0.7, 's':40, 'edgecolor': 'k'},
                     height = 2.8
                     )  #연속확률변수 그래프
    g.fig.suptitle(title, fontsize=18, weight='bold')
    g.fig.subplots_adjust(top=0.9)  # Adjust title position)

    # 2) 파일 저장 (fname이 None이 아니면)
    if fname is not None:
        g.fig.savefig(fname, dpi=200)

        plt.close(g.fig)  # free memory

    # 3) ‘KMeans’ 제목으로 호출된 순간, 세 장을 합쳐서 보여 주도록함
    if title.lower().startswith('kmeans'):
        # 각각 저장된 이미지 파일 경로를 미리 지정해 두었기 때문에, 경로가 모두 존재한다는 전제아래 작동
        imgs = []
        files = ['original.png', 'em.png', 'kmeans.png']
        for f in files:
            try:
                imgs.append(mpimg.imread(f))
            except FileNotFoundError:
                # 파일없다면 무시
                pass

        if len(imgs) == 3:
            # 3장을 가로로 붙여서 하나의 큰 fig로 생성
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            titles = ['Original', 'EM', 'KMeans']
            for ax, t, im in zip(axes, titles, imgs):
                ax.imshow(im)
                ax.set_title(t, fontsize=14, weight='bold')
                ax.axis('off')
            fig.tight_layout()
            # 'combined.png'로 저장하고 화면에 띄움
            fig.savefig('combined.png', dpi=200)
            plt.show()
            plt.close(fig)

# ----------------------------- Main Execution -----------------------------
if __name__ == '__main__':
    # Loading and labeling data
    iris = datasets.load_iris()
    original_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                 columns=iris['feature_names'] + ['labels'])
    original_data['labels'] = original_data['labels'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    plotting(original_data, title='Original Data', fname='original.png')

    # Only data is used W/O labels because EM and KMeans are unsupervised learning
    X_scaled = iris['data']  # 표준화가 필요 없다면 그대로 사용
    data = X_scaled

    # Unsupervised learning(clustering) using EM algorithm
    EM_model = EM(n_clusters=3, iteration=iteration)
    # print("EM initialization started")        # DEBUG
    EM_pred = EM_model.fit(data)
    # print("EM finished")      # DEBUG
    #print(np.unique(EM_pred, return_counts=True))     #DEBUG
    EM_pd = pd.DataFrame(data=np.c_[data, EM_pred], columns=iris['feature_names'] + ['labels'])
    EM_pd['labels'] = EM_pd['labels'].astype(str)  # string으로 변환하여 seaborn에서 색깔 구분
    plotting(EM_pd, title='EM Clustering Result', fname='em.png')

    # Why are these two elements almost the same? Write down the reason in your report. Additional 10 points
    print(f'pi :            {EM_model.pi}')
    print(f'count / total : {np.bincount(EM_pred) / 150}')

    # Unsupervised learning(clustering) using KMeans algorithm
    KM_model = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data)
    KM_pred = KM_model.predict(data)
    KM_pd = pd.DataFrame(data=np.c_[data, KM_pred], columns=iris['feature_names'] + ['labels'])
    KM_pd['labels'] = KM_pd['labels'].astype(str)
    plotting(KM_pd, title='KMeans Clustering Result', fname='kmeans.png')

    # No need to explain.
    for idx in range(2):
        EM_point = np.argmax(np.bincount(EM_pred[idx * 50:(idx + 1) * 50]))
        KM_point = np.argmax(np.bincount(KM_pred[idx * 50:(idx + 1) * 50]))
        EM_pred = np.where(EM_pred == idx, 3, EM_pred)
        EM_pred = np.where(EM_pred == EM_point, idx, EM_pred)
        EM_pred = np.where(EM_pred == 3, EM_point, EM_pred)
        KM_pred = np.where(KM_pred == idx, 3, KM_pred)
        KM_pred = np.where(KM_pred == KM_point, idx, KM_pred)
        KM_pred = np.where(KM_pred == 3, KM_point, KM_pred)

    EM_hit = np.sum(iris['target'] == EM_pred)
    KM_hit = np.sum(iris['target'] == KM_pred)
    print(f'EM Accuracy: {round(EM_hit / 150, 2)}    Hit: {EM_hit} / 150')
    print(f'KM Accuracy: {round(KM_hit / 150, 2)}    Hit: {KM_hit} / 150')
