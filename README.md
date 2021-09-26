# The-2nd-Computer-Vision-Learning-Contest
[제 2회 컴퓨터비전 EMNIST 분류 경진대회](https://dacon.io/competitions/official/235697/leaderboard)

### Private : 43/216(19%)

## 1. 주제
- 합성한 MNIST 이미지 속에 들어있는 알파벳 찾기

## 2. 배경

- 손글씨 이미지인 MNIST 데이터 세트는 이 분야의 고전적인 문제로 잘 알려져 있음.
- 무작위로 합성된 10 ~ 15개의 글자를 분류하는 multi-label classification

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
