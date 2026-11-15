# 데이터 및 선행 조사 (2026-11-15)

> 프로젝트 착수 전 데이터 가용성과 선행 연구를 확인한 결과. 출처는 3절.

## 1. 확인 결과

| # | 확인 항목 | 결과 | 상태 |
|---|---|---|---|
| 1 | 공개 배치 벤치마크가 있다 | **있음.** MCNC(ami33, ami49 등 9~49블록), GSRC(n10~n300) — 블록·연결(nets)·배치 파일. UMich 사이트 배포. **전력값은 없음** | 확인 |
| 2 | 최신 대규모 데이터 | **FloorSet** (Intel Labs, ICCAD 2024): 학습 샘플 1M급, CC BY 4.0, **35GB**, 파티션 20~120개, **전력값 언급 없음** | 확인 (1단계엔 과함) |
| 3 | 블록 전력값을 구할 수 있다 | **HotSpot 예제** `ev6.flp`(Alpha 21264 단일 코어 평면도) + `gcc.ptrace`(전력 추적)에 있음. 단 **블록 연결 정보는 없음** | 확인 |
| 4 | 공개 열해석 도구(HotSpot)를 쓸 수 있다 | 공개 v6.0, 리눅스용 C 프로그램 → 본 프로젝트에서는 미사용 | 조건부 |
| 5 | 구글 강화학습 칩 배치 논란 | **사실.** Nature 2021 논문 → UCSD Cheng·Kahng 팀 재구현(ISPD 2023) "기준 방법보다 낫지 않음", Markov "The False Dawn"(2023), 구글 반박 "That Chip Has Sailed"(2024). Kahng의 2021 Nature 해설 기사는 2023년 철회(논문 자체 철회는 확인 못 함) | 확인 |
| 6 | AI 빠른 열 예측 연구가 있다 | **이미 많음.** ThermEDGe(CNN, 전력 지도→온도 지도, 밀리초), DeepOHeat(3D 칩, 1000~300000배 가속 보고), GCN 열지도(HotSpot 대비 RMSE 0.31K 보고), HBM 칩렛 대리모델(2025) | 확인 |
| 7 | 상용 도구도 AI 열해석으로 간다 | **사실.** Cadence Celsius Studio(2024.1 발표): AI 열 플랫폼, 2.5D/3D 패키지, 열응력 휨까지, "최대 10배 빠름" 주장 | 확인 |
| 8 | 칩 온도 센서 배치 연구가 있다 | **있음.** 배치(핫스팟 기반, 기울기 기반, 엔트로피 기반, EigenMaps)와 복원(보간, 압축센싱, 신경망 — RealMaps는 인텔 칩에서 약 0.9°C 오차 보고) 두 갈래 | 확인 |
| 9 | 레이저 어시스트 본딩 열·휨 해석 | **있음.** 기판이 칩보다 덜 뜨거워 열팽창 차이·휨이 줄어듦, 대형 칩에서 휨 약 50% 감소 보고, KAIST·서울과기대 해석 논문(적외선 측정으로 검증한 사례 포함) | 확인 |
| 10 | 개인 노트북으로 계산 가능 | Python 3.14, NumPy 2.3.5, SciPy 1.16.3, PyTorch 2.14(CPU). **GPU 없음** → 작은 CNN은 CPU로 가능, 큰 모델은 느림 | 확인 |

## 2. 프로젝트 설계에 반영한 점

1. **새로움:** "AI로 빠른 열 예측"은 연구(6)와 상용(7) 모두 이미 있음 → "새 발명"이 아니라 "업계가 가는 방향을 작게 직접 만들어 확인"으로 표현해야 함
2. **AI 주장:** 구글 논란(5) 때문에 기준 방법(시뮬레이티드 어닐링) 비교는 필수, "AI가 이겼다"는 숫자로만
3. **데이터:** 연결 정보와 전력값을 둘 다 가진 공개 데이터가 없음 → 하나는 가정해야 함
4. **검증:** HotSpot 비교에는 리눅스 환경이 필요 → 해석해·손계산 검증으로 대체
5. **센서(1.5단계):** 연구가 정리돼 있어 기준 방법(보간 등)을 바로 잡을 수 있음
6. **레이저 본딩(Stage 3 후보):** 국내 해석 논문이 있어 비교 기준으로 활용 가능

## 3. 출처
- HotSpot HOWTO: https://lava.cs.virginia.edu/HotSpot/HotSpot-HOWTO.htm
- UMich 배치 벤치마크: https://vlsicad.eecs.umich.edu/BK/FPUtils/
- FloorSet: https://arxiv.org/abs/2405.05480 · https://huggingface.co/datasets/IntelLabs/FloorSet
- 구글 논란: https://spectrum.ieee.org/amp/chip-design-controversy-2659708634 · https://arxiv.deeppaper.ai/papers/2306.09633v10 · https://arxiv.deeppaper.ai/papers/2411.10053v1 · https://www.nature.com/articles/d41586-021-01515-9.pdf
- AI 열 예측: https://arxiv.org/pdf/2110.14197 (ThermEDGe 계열) · https://arxiv.org/html/2503.04049v1 (HBM 대리모델)
- Cadence Celsius Studio: https://www.businesswire.com/news/home/20240131987413/en/
- 센서 배치·복원: https://infoscience.epfl.ch/record/174866 · https://arxiv.org/pdf/2601.06740 · https://publications.polymtl.ca/54784/1/2023_Touati_Accurate_On-Chip_Thermal_Peak_Detection.pdf
- 레이저 본딩: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10051732/ · https://koasas.kaist.ac.kr/handle/10203/312714 · https://semiengineering.com/lab-flip-chip-reflow-process-robustness-prediction-by-thermal-simulation

※ 검색 결과 요약 기반이며, 논문 전문은 대부분 확인하지 않았다. 수치 인용 시 원문 확인이 필요하다.

## 4. ami33 파일 점검
- 출처: https://vlsicad.eecs.umich.edu/BK/MCNCbench/HARD/ (ami33.blocks 2,697B / ami33.nets 13,891B / ami33.pl 922B, 텍스트)
- 블록 33개(모두 직사각형 고정 크기), 외부 단자(패드) 42개(위치 있음), 넷 123개
- 핀 수: 파일 머리말은 522, 넷별 선언 차수 합은 520 → **머리말 숫자가 틀림**, 파서는 520으로 일치
- 전원 넷(GND·VDD) 2개, 최대 차수 56 → 배선 길이에 넣으면 값이 크게 부풀 수 있음
- 블록 면적 합 1,156,449 (정사각형 한 변 약 1,075 단위), 패드 좌표는 원래 레이아웃 기준(x 최대 2,058)
- 단위는 파일에 없음 → 10 mm 칩에 비율로 맞춰 사용
- 전력값 없음 → 가정값 사용 ([`criteria.md`](criteria.md) #11)
