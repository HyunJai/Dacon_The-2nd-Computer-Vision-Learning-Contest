# 카메라 이미지 품질 향상 AI 경진대회
[LG AI Research 경진대회](https://dacon.io/competitions/official/235746/overview/description)

### Private : 18/235(8%)

## 1. 주제
- 🌟빛 번짐으로 저하된 📷카메라 이미지 품질을 향상시키는 AI 모델 개발

## 2. 배경

- 빛이 렌즈를 통해 사진이 찍힐 때 물리적인 요인에 의해 다양한 빛 번짐이 발생함
- 이러한 물리적 빛 번짐은 반도체 기술만으로 처리하기 어려움
- AI 기술로 문제를 적극적으로 해결하고자

## 3. 주최 및 주관

- 주최 : ﻿LG AI Research
- 주관 : 데이콘

## 4. 모델
- EfficientB1 + UNet(Segmentation Model)
- Encoder Weights : 이미지넷(ImageNet)

## 5. 데이터 증강
- Horizontal Flip

## 6. 파라미터
- Epoch : 200번
- Drop rate : 0.1
- Optimizer : Adam(1e-4)
- Loss : L1 함수 + Sigmoid

## 7. 학습시 GPU 최적화
- NVIDIA Apex : opt level 01, amp.initialize() + amp.scale_loss
- torch.distributed.init_process_group()
- DataLoader : pin_memory = True, num_workers * GPU 갯수
- torch.cuda.empty_cache() : validation 진행 전 메모리 초기화

## 💡 본 경진대회에서 제안한 방법
1. parameter 학습을 효율적으로 하기위해 Efficient B1과 UNet를 결합한 pretrained 모델 사용함
2. 모댈의 input 이미지의 크기를 patch 단위로 커팅함
3. 모델의 input 이미지에 대해 Sigmoid를 적용함
4. 마지막에 inference시 patch 단위의 이미지를 reconstruction 시켜 psnr의 성능을 높임
