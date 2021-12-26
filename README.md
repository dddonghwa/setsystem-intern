# 딥러닝 기반 선박 및 파랑 탐지

- 참고 자료 : 한요섭님 [github](https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet) 및 [Youtube : U-Net 실습](https://www.youtube.com/watch?v=sSxdQq9CCx0)
- _회사 보안 상 훈련을 위한 데이터셋은 업로드하지 않았습니다_

## 목표
RDM(Range Doppler Map) 상에서 선박 및 파랑(브래그 신호 영역) 탐지

<img width="50%" alt="Compact HF Surface Wave Radar Data Generating Simulator for Ship Detection and Tracking" src="https://github.com/dddonghwa/setsystem-intern/blob/main/image/image1.png">

(출처 : "Compact HF Surface Wave Radar Data Generating Simulator for Ship Detection and Tracking" [Google Scholar](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Compact+HF+Surface+Wave+Radar+Data+Generating+Simulator+for+Ship+Detection+and+Tracking&btnG=))
- 기존의 접근법 : Detectron2의 Mask R-CNN을 transfer learning 시킨 모델 구현 및 적용
- 목표 : 
1) 최신 모델 적용을 통한 선박 탐지 성능 (정확도, 속도) 개선 
2) 실시간 탐지 가능성 확인
- 적용 가능 task 
1) __객체 탐지__(Object Detection) : 물체가 어떤 분류에 속하고, 이미지 상에 어디에 위치하는지 박스형 경계로 예측
2) __이미지/의미론적 분할__(Image Segmentation) : 픽셀 단위의 분류

## 객체 탐지 vs 이미지 분할
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image2.png' width='70%'>

## 적용 가능 모델
#### 객체 탐지
1. Two Stage 
- R-CNN(14) → Fast R-CNN(15) → Faster R-CNN(15) → __Mask R-CNN__ (17)
- FPN(Feature Pyramid Net, 17)
2. One Stage 
- YOLO (You Only Live Once, 16) → YOLOv2(17) → YOLOv3(18) → __YOLOv4__(20)
- SSD (Single Shot Multibox Detector, 2016) → RefineDet(18)
3. Multi Stage
- Cascade R-CNN(17) → Cascade Mask R-CNN(17)
- __HTC__(Hybrid Task Cascade, 19) 
- RetinaNet (21)

#### 이미지 분할
- Mask R-CNN
- FCN (Fully Convolution Network)
- U-Net 
- DeepLabv3+
- PSPNet
- HRNetV2


## Detectron2 vs DeepLabv3+ vs U-Net 비교
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image5.png' >

## DeepLabv3+ 실패 원인
- Detectron2에서 구현 가능할 것으로 예상했지만 Detectron2의 메인 수행 모델이 아니고 서브 프로젝트에서 개발하고 있는 모델이라서 사용자화시켜서 구현하기엔 정보의 한계가 있었음 
- tensorflow 공식 github에 나온 deeplabv3+ 소스코드 : tensorflow 버전 1.X를 기반으로 하기 때문에 현재의 버전 2.X과 충돌


## Detectron2 vs U-Net 구현 결과 비교
<img src='https://github.com/dddonghwa/setsystem-intern/blob/main/image/image6.png' >

