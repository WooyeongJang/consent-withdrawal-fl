"""resnet1d_wang (Strodthoff et al., 2021의 구조를 논문·문서 기준으로 새로 작성).

구성(벤치마크 기본값):
  앞단  Conv1d(12→128, k=7, s=1) → BN → ReLU (풀링 없음)
  몸통  BasicBlock ×3, 채널 128 고정, 커널 [5, 3], 2·3번째 블록 stride 2(1×1 합성곱+BN 지름길)
  출력부 concat pooling[max, avg] → BN → Dropout 0.25 → Linear(256→128) → ReLU → BN → Dropout 0.5 → Linear(128→5)

max_pool="amax"는 출력부 최대 풀링을 torch.amax로 계산한다. 출력과 기울기가 AdaptiveMaxPool1d와 같고
(reports/2026-10-25_벤치마크_원구현_검토.md), GPU 결정적 모드에서 실행된다.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, ch, stride, kernels=(5, 3)):
        super().__init__()
        k1, k2 = kernels
        self.conv1 = nn.Conv1d(ch, ch, k1, stride=stride, padding=(k1 - 1) // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k2, padding=(k2 - 1) // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.shortcut = (nn.Sequential(nn.Conv1d(ch, ch, 1, stride=stride, bias=False), nn.BatchNorm1d(ch))
                         if stride != 1 else nn.Identity())

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ConcatPool(nn.Module):
    def __init__(self, max_pool="adaptive"):
        super().__init__()
        self.max_pool = max_pool
        self.mp = nn.AdaptiveMaxPool1d(1)
        self.ap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        mx = torch.amax(x, -1, keepdim=True) if self.max_pool == "amax" else self.mp(x)
        return torch.cat([mx, self.ap(x)], 1)


class ResNet1dWang(nn.Module):
    def __init__(self, n_classes=5, in_ch=12, ch=128, max_pool="adaptive"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, ch, 7, padding=3, bias=False), nn.BatchNorm1d(ch), nn.ReLU())
        self.body = nn.Sequential(BasicBlock(ch, 1), BasicBlock(ch, 2), BasicBlock(ch, 2))
        self.head = nn.Sequential(
            ConcatPool(max_pool), nn.Flatten(),
            nn.BatchNorm1d(2 * ch), nn.Dropout(0.25), nn.Linear(2 * ch, 128), nn.ReLU(),
            nn.BatchNorm1d(128), nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))
