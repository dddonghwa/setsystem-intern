## Detectron2 폴더 설명
사용한 내용만
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
