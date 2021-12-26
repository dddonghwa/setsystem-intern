# U-Net 기반 파랑  

### 참고 자료
- 한요섭님 [깃허브](https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet), [유투브 : U-Net 실습](https://www.youtube.com/watch?v=sSxdQq9CCx0)

## 폴더 설명
1. datasets/ 
CSQ 데이터셋 포함한 폴더
- train : 훈련 데이터 (100개 * json,png 2개씩 = 200개)
- val : 검증 데이터 (5개*2 = 10개) 
- test : 테스트 데이터, eval.py 실행 시 해당 폴더 안의 데이터를 사용
- example : 한요섭님의 U-Net 실습에 필요한 샘플 데이터 (세포 이미지)

2. labelme/
CSQ 데이터셋 전처리를 위한 폴더
전처리 결과로 얻어낸 results/input 및 label 데이터들을 ../datasets/train,test,val 폴더에 직접 옮겨준다.
- data/ : json, png 형식의 CSQ 파일
- json2npy.py : 기존에 json, png 파일로 있던 데이터를 npy파일로 바꿔주는 코드, 결과는 results 폴더에 생성된다.
- results/ : 결과 폴더
	- input/ : 입력 이미지 gray-scale로 변환한 뒤 npy 파일로 저장한 데이터
	- label/ : 라벨링된 json 파일을 픽셀 단위로 레이블링한 후 npy 파일로 저장한 데이터
	- visual/ : 시각화 자료
	- class_names.txt : json 파일에 레이블링된 클래스 종류 및 할당 숫자
- labelme2voc.py : json2npy.py의 원본 [github 출처](https://github.com/wkentaro/labelme/tree/main/examples/instance_segmentation/labelme2voc.py)
- labels.txt : json 파일에 레이블링된 클래스 이름, 직접 수정 필요


3. checkpoint/
50 epoch마다 학습한 모델 저장하는 폴더
이 폴더 안에 있는 모델을 load해서 test를 수행한다.

4. log/
학습하면서 생성된 로그를 저장하는 폴더

5. results/
eval.py를 실행하면 학습한 모델이 test 데이터셋에 대하여 에측한 결과를 저장하는 폴더
- numpy/, png/ : 예측 결과가 각각 .npy, .png 형식으로 저장된다.
- combined.png, Detectron2.png, U-Net.png : 입력 이미지와 레이블(정답지), 예측 결과를 시각화한 플롯, miou.py를 수행하면 생성된다.

6. test-rgb/
combined.png, Detectron2.png, U-Net.png을 생성하기 위해서 모델의 test 이미지(.png)만 저장한 폴더

7. 나머지 파일
- util.py : train.py에 사용되는 함수/모듈을 정의하는 코드 (수정X)
- train.py : 모델 훈련하는 코드
- run_unet.ipynb : colab에서 tensorboard로 훈련 확인하기 위한 주피터 노트북
- model.py : 모델 정의한 코드
- miou.py : Detectron2 와 U-Net의 mIoU 결과 비교하기 위한 코드, 계산된 mIoU를 출력하고 results/ 폴더에 플롯을 생성한다.
- eval.py : 훈련한 모델을 test 데이터셋에 적용시켜 예측 결과를 산출해내는 코드
- display_results.py : 입력 이미지와 레이블, 예측 결과를 임의의 한 샘플만 골라서 플롯으로 그려주는 코드
- dataset.py 훈련을 위한 데이터로더(dataloader), 트랜스폼(transform) 정의하는 코드
- data_read.py :  U-Net 실습 중 tiff로 되어있던 이미지를 전처리하는 코드



### Detectron2 폴더 사용한 내용만
* 폴더 위치 : /home/set-spica/Desktop/test_jh/jihye

1. HFradar Segmentation /
- old_sample : 2018.10.16 HFradar 관측 자료인 CSQ 데이터
	- train : 훈련 데이터 105개*png,json 2개씩 = 210개
	- test : 테스트 데이터 5개*2 = 10개
- new_sample(사용x) : 2019.10.01 HFradar 관측 자료인 CSQ 데이터
	- train : 훈련 데이터 133개*png,json 2개씩 = 266개
	- test : 테스트 데이터 22개*2 = 44개
	

2. output /
모델 학습 결과 저장

3. npy/
U-Net 수행 결과와 비교하기 위해 json2npy.py 실행시키면 해당 폴더에 테스트 이미지(png), 레이블, 예측 결과가 npy 파일로 저장된다.

4. 나머지 파일들
- HFR_prediction.py : 학습한 모델을 테스트 데이터에 적용하여 예측 결과를 시각화하는 코드 (시각화만 하고 예측 결과 출력은 안해줌)
- json2npy.py : 테스트 데이터 레이블과 Detectron2 모델로 예측한 결과를 npy로 저장시키는 코드
- labels.txt : 모델 입럭 데이터에 레이블링된 클래스 종류, 직접 수정해주어야 함
- train_code.py : Detectron2에서 미리 학습시킨 mask_rcnn_R_50_FPN_3x.yaml 파일을 불러들어 CSQ 데이터를 추가로 학습시키고 모델 저장
