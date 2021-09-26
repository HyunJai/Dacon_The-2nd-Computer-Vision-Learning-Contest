# 카메라 이미지 품질 향상 AI 경진대회
[LG AI Research 경진대회](https://dacon.io/competitions/official/235746/overview/description)

### Private : 18/235(7%)

## 1. 주제
- 🌟빛 번짐으로 저하된 📷카메라 이미지 품질을 향상시키는 AI 모델 개발

## 2. 배경

- 빛이 렌즈를 통해 사진이 찍힐 때 물리적인 요인에 의해 다양한 빛 번짐이 발생함
- 이러한 물리적 빛 번짐은 반도체 기술만으로 처리하기 어려움
- AI 기술로 문제를 적극적으로 해결하고자

## 3. 대회 설명

- (256, 256)크기의 이미지 속에 10 ~ 14개의 글자(알파벳 a – Z, 중복 제외)가 무작위로 배치되어 있음.
- 이번 대회의 문제는 이러한 이미지 속에 들어있는 알파벳과 들어있지 않은 알파벳을 분류하는 multi-label classification

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
