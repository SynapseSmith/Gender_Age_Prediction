### 디렉터리 구조

```
Project/
│
├── config/
│   └── hyper_parameters.yaml
│ 
├── data/
│   ├── Training
│   ├── Validation
│   ├── Test
│   ├── Denoised_Training
│   ├── Denoised_Validation
│   ├── Denoised_Test
│   ├── Combined_Training
│   ├── Combined_Validation
│   ├── Combined_Test
│ 
├── etl/
│   ├── extract_labels.py
│   ├── load_data.py
│   ├── preprocess_image.py
│
├── models/
│   ├── efficientnetb0.py
│   ├── gender_age_model.py
│
├── scripts/
│   ├── eval_inferences.py
│   ├── eval_model.py
│   ├── grad_cam.py
│   ├── infer_image.py
│   ├── infer_images.py
│   ├── train_model.py
│   └── denoise/
│       ├── save_denoised_images.py
│       ├── test_denoising.py
│       ├── train_auto_encoder.py
│
└── utils/
    ├── count_ages.py
    ├── count_genders.py
    ├── extract_labels.py
    ├── face_dataset.py
    ├── preprocess_image.py
    ├── rename_denoised_file.py
    └── split_test_image.py
```

#### 1. `config/`
- **hyper_parameters.yaml**: 모델 학습 및 평가를 위한 하이퍼파라미터를 포함한 파일.

#### 2. `data/`
- **Training/**: 원본 훈련 데이터 (노이즈 포함)
- **Validation/**: 원본 검증 데이터 (노이즈 포함)
- **Test/**: 훈련 데이터로부터 분할한 원본 테스트 데이터 (노이즈 포함)
- **Denoised_Training**: 노이즈를 제거한 훈련 데이터
- **Denoised_Validation**: 노이즈를 제거한 검증 데이터
- **Denoised_Test**: 노이즈를 제거한 테스트 데이터
- **Combined_Training**: Training + Test + Denoised_Training + Denoised_Test
- **Combined_Validation**: Validation + Denoised_Validation
- **Combined_Test**: Test + Denoised_Test

#### 3. `etl/` (denoise 폴더의 파일들에서 사용하는 함수들)
- **extract_labels.py**: 데이터셋에서 레이블을 추출하는 스크립트.
- **load_data.py**: 데이터 로드 및 처리, 모델 학습을 위한 스크립트.
- **preprocess_image.py**: 이미지 전처리 스크립트.

#### 4. `models/`
- **efficientnetb0.py**: EfficientNetB0 모델 아키텍처 정의.
- **gender_age_model.py**: 여러 개의 사전 학습된 모델 정의.

#### 5. `scripts/`
- **eval_inferences.py**: 모델 추론을 성별을 잘못 분류한 경우, 나이 예측 오차가 10살 이상인 경우, 예측 시간을 통해 평가하는 스크립트.
- **eval_model.py**: 학습된 모델의 성별 정확도, f1 score, 나이의 MAE을 평가하는 스크립트.
- **grad_cam.py**: 모델 예측을 해석하기 위해 Grad-CAM 시각화를 생성하는 스크립트.
- **infer_image.py**: 단일 이미지에 대한 추론하는 스크립트.
- **infer_images.py**: 여러 이미지에 대한 추론하는 스크립트.
- **train_model.py**: 모델을 학습시키는 스크립트.
- **denoise/**: 이미지 노이즈 제거와 관련된 스크립트를 포함한 디렉터리.
  - **save_denoised_images.py**: 노이즈가 제거된 이미지를 저장.
  - **test_denoising.py**: 모델의 노이즈 제거 성능을 시각화를 통해 테스트.
  - **train_auto_encoder.py**: 노이즈 제거를 위한 오토인코더 학습.

#### 6. `utils/`
- **count_ages.py**: 데이터셋의 나이 분포를 시각화하는 유틸리티 스크립트.
- **count_genders.py**: 데이터셋의 성별 분포를 시각화하는 유틸리티 스크립트.
- **extract_labels.py**: 파일 이름으로부터 레이블을 추출하는 유틸리티 스크립트.
- **face_dataset.py**: 얼굴 데이터셋 작업을 처리하는 유틸리티 스크립트.
- **preprocess_image.py**: 이미지 전처리 유틸리티 스크립트.
- **rename_denoised_file.py**: 노이즈 제거된 파일 이름에 'denoised'를 추가하는 스크립트.
- **split_test_image.py**: 모니터링을 위해 테스트 이미지를 여러 세트로 분할하는 스크립트.# Gender_Age_Prediction
# Gender_Age_Prediction
