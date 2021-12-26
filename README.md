# 딥러닝 기반 선박 및 파랑 탐지

- 참고 자료 : 한요섭님 [github](https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet) 및 [Youtube : U-Net 실습](https://www.youtube.com/watch?v=sSxdQq9CCx0)
- _회사 보안 상 훈련을 위한 데이터셋은 업로드하지 않았습니다_

## 목표
RDM(Range Doppler Map) 상에서 선박 및 파랑(브래그 신호 영역) 탐지
<img width="714" alt="Compact HF Surface Wave Radar Data Generating Simulator for Ship Detection and Tracking" src="https://github.com/dddonghwa/setsystem-intern/blob/main/image/image1.png">

(출처 : "Compact HF Surface Wave Radar Data Generating Simulator for Ship Detection and Tracking" [Google Scholar](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Compact+HF+Surface+Wave+Radar+Data+Generating+Simulator+for+Ship+Detection+and+Tracking&btnG=))
- 기존의 접근법 : Detectron2의 Mask R-CNN을 transfer learning 시킨 모델 구현 및 적용
- 향후 개선 사항 : 1) 최신 모델 적용을 통한 선박 탐지 성능 (정확도, 속도) 개선 2) 실시간 탐지 가능성 확인
- 적용 가능 task : __객체 탐지__(Object Detection), __이미지/의미론적 분할__(Image Segmentation)
	- 객체 탐지 : 물체가 어떤 분류에 속하고, 이미지 상에 어디에 위치하는지 박스형 경계로 예측
	- 이미지/의미론적 분할 : 픽셀 단위의 분류

## 객체 탐지 vs 이미지 분할
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image2.png' width='80%'>

## 적용 가능 모델
1. 객체 탐지
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image4.png'>

2. 이미지 분할
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image3.png' height=150>

## Detectron2 vs DeepLabv3+ vs U-Net 비교
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image5.png' width='80%'>

## DeepLabv3+ 실패 원인
- Detectron2에서 구현 가능할 것으로 예상했지만 Detectron2의 메인 수행 모델이 아니고 서브 프로젝트에서 개발하고 있는 모델이라서 사용자화시켜서 구현하기엔 정보의 한계가 있었음 
- tensorflow 공식 github에 나온 deeplabv3+ 소스코드 : tensorflow 버전 1.X를 기반으로 하기 때문에 현재의 버전 2.X과 충돌


## Detectron2 vs U-Net 결과 비교
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image6.png' width='80%'>

### ppt 폴더
발표 자료
- 1차 발표 : 과제 정의, 객체 탐지(Object Detection) 및 의미론적 분할(Semantic Segmentation) 모델 개요, 객체 탐지 vs 의미론적 분할 비교
- 2차 발표 : Detectron2 vs DeepLabv3+ vs U-Net 비교
- 최종 발표 : Deeplabv3+ 구현 실패 원인 분석, U-Net 구현 과정 및 결과, Dectectron2(Mask R-CNN) 구현 과정 및 결과
