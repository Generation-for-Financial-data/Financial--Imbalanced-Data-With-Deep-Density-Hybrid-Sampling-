# Financial--Imbalanced-Data-With-Deep-Density-Hybrid-Sampling-

imblanced 상황에서 Hybrid sampling 방법론 중 하나인 
DDHS 논문의 아키텍쳐를 기반으로 금융 데이터에 맞게 contribution을 추가하여 구현 

## <아키텍쳐>
(실험을 통해 변경가능)

1) 연속형, 범주형 판단
2) small, big class에 따라 분리
3) 각 class에 따로 연속형만 GMM(k=2) fitting 후 상위 N % 만 Filtering
4) ctgan으로 연속형 변수만 생성 후 Density 기반의 적합성 판단
5) 생성된 연속형 변수와 cosine similarity 기반의 가장 가까운 기존 변수의 범주형값 카피
