# 분자 독성 예측 — 모델 점수에서 평가 신뢰성으로

수업에서 제공한 분자 Fingerprint·물성 데이터로 독성을 분류한 프로젝트입니다. 기존 LightGBM 모델 방향을 유지하면서 **같은 분자의 중복, 표현 충돌, 골격 간 중복과 임계값 선택의 영향을 점검하고 재현 가능한 평가 코드로 정리**했습니다.

**라벨은 `0=독성`, `1=비독성`입니다.** 공식 F1은 비독성(1)을 기준으로 하며 독성 탐지 성능과 다릅니다. 이 프로젝트는 교육 데이터의 후향적 분석이며 실제 신약 안전성 판정 도구가 아닙니다.

## 무엇을 마무리했나

| 기존 수업·제출 실험 | 개인 고도화 |
|---|---|
| LightGBM, Morgan, Morgan8, 앙상블 비교 | 주 모델을 Original LightGBM으로 미리 고정, Logistic Regression 기준 모델 비교 |
| 무작위·골격 분할과 OOF 임계값 실험 | 바깥 5-fold 평가 / 안쪽 3-fold 임계값·반복 횟수 결정 |
| 전체 데이터의 상수 열 49개 제거 | 상수 제거·스케일링을 각 학습 fold 안에서 fit |
| SMILES 문자열 중복 확인 | canonical stereo·connectivity·동일 수치 입력·골격 점검 |
| 리더보드 공식 F1 위주 | 독성 F1·재현율·독성 누락 수·오류 구간을 함께 보고 |
| 노트북 중심, 실행 환경 미고정 | Python 모듈·버전 고정·CLI·자동 검사·집계 보고서 |

## 이번에 확인한 데이터 문제

학습 데이터는 8,349행, 수치 특성 3,076개입니다. ECFP·FCFP·PTFP 각각 1,024개와 `MolWt`, `clogp`, `sa_score`, `qed`를 사용합니다. 독성 3,807행, 비독성 4,542행입니다.

- 원문 SMILES 중복은 없지만 canonical stereochemical SMILES가 같은 분자 2행이 있습니다. 두 행의 라벨 충돌은 없습니다.
- 동일 수치 입력 241개 그룹·534행 중 **33개 그룹·73행은 라벨이 다릅니다.** 입체 정보 표현 손실 등 여러 원인이 가능하므로 오류라고 단정해 삭제하지 않았습니다.
- 입체 정보를 제외한 연결 구조에서는 라벨 충돌 34개 그룹·75행을 확인했습니다.
- 예측 입력 927행 중 학습 데이터와 canonical stereo identity가 같은 행은 1개, connectivity가 같은 행은 81개, **동일 수치 입력은 82개**, 같은 골격은 627개입니다. 이 수치는 서로 겹치므로 합산하지 않습니다.
- 상수 특성 49개와 물성 IQR 이상치는 기술 통계로 보고하며, 이상치를 자동 삭제하지 않습니다.

분할 그룹은 **chirality-free Bemis–Murcko 골격 또는 동일 수치 입력을 공유하는 관계의 전이적 합집합**입니다. 골격 4,161개가 합집합 4,159개 그룹이 됐고 최대 그룹은 92행입니다. 고리가 없는 2행은 연결 구조로 구분합니다. 염·호변이성질체 정규화나 모든 유사 분자 간 분리를 보장하지는 않습니다.

## 평가 설계

1. 기존 실험을 근거로 Original LightGBM을 주 모델로 미리 정합니다. Morgan8의 전체 label 기반 선택 특성은 사용하지 않습니다.
2. 그룹을 공유하지 않는 바깥 5-fold를 만듭니다. 각 행은 바깥 검증에 한 번만 등장합니다.
3. 각 바깥 학습 데이터 안에서 그룹 기반 안쪽 3-fold를 구성합니다. 각 학습 fold에서만 상수 제거를 fit하며 Logistic Regression은 스케일링도 학습 데이터에서만 fit합니다.
4. 안쪽 LightGBM 검증으로 조기 종료하고 반복 횟수의 중앙값을 정합니다. 안쪽 OOF 예측으로 비독성 F1 임계값을 선택합니다. 탐색 범위는 0.20~0.80, 간격 0.005이며 동점은 0.5에 가까운 값, 이후 작은 값을 선택합니다.
5. 결정한 반복 횟수로 바깥 학습 데이터를 재학습합니다. **바깥 검증은 조기 종료·임계값·설정 선택에 사용하지 않습니다.**
6. 바깥 OOF 결과를 합쳐 평가합니다. 결과를 보고 주 모델을 교체하거나 재튜닝하지 않았습니다.
7. 이후 전체 데이터의 별도 안쪽 CV로 최종 아티팩트 정책을 정하고 전체 데이터에 재학습합니다. 바깥 OOF는 이 전체 데이터 학습 모델 자체를 독립 평가한 결과가 아닙니다.

안쪽 OOF는 반복 횟수·임계값 선택용이며 독립 평가로 제시하지 않습니다. fold 점수의 표준편차는 기술 통계이고 신뢰구간이 아닙니다. 이미 과거에 탐색한 데이터이므로 이번 중첩 평가도 **새 독립 테스트가 아닌 후향적 검증**입니다.

## 결과: 공식 점수와 독성 탐지는 다르다

[이번 실행의 전체 집계](reports/retrospective-v1.json) / [결과 읽기 노트북](03_reproducible_evaluation.ipynb)

| 바깥 OOF | 비독성 F1(1) | 독성 F1(0) | 독성 재현율(0) | ROC-AUC |
|---|---:|---:|---:|---:|
| LightGBM, 안쪽 F1 선택 임계값 | **0.7916** | 0.6743 | 0.5771 | **0.8420** |
| Logistic Regression, 안쪽 F1 선택 임계값 | 0.7668 | 0.6337 | 0.5406 | 0.8035 |
| LightGBM, 고정 임계값 0.5 비교 | 0.7879 | **0.7282** | **0.7000** | 0.8420 |

F1 선택 정책에서 독성 3,807건 중 **1,610건이 비독성으로 예측**됐고, 고정 0.5에서는 1,142건이었습니다. 공식 비독성 F1의 소폭 개선이 독성 탐지에 유리한 것은 아닙니다. 0.5는 사전에 정한 비교 기준이며, 이번 결과를 보고 최종 정책을 자동 변경하지 않았습니다. 확률 보정·실제 안전성 요구 재현율·오류 비용은 검증하지 않았습니다.

분자량이 바깥 학습 데이터의 75% 분위수를 넘는 구간도 따로 집계했습니다. 사후 오류 분석이며 특정 물성의 인과 효과나 우수성을 주장하지 않습니다. 최종 모델의 gain 상위 20개는 기술 통계일 뿐 화학 기전 설명이 아니며, 해시 Fingerprint 비트를 하나의 화학 작용기로 단정하지 않습니다.

### 과거 리더보드는 별도 기록

과거 4차 제출의 공식 F1은 **0.8438**, Accuracy 0.8263, Precision 0.8086, Recall 0.8824였습니다. [본인 제출 기록만 집계](reports/historical-leaderboard.json)했고 다른 수강생 정보·PDF는 업로드하지 않았습니다. **4차 제출 파일과 최종 모델 설정의 대응은 확인되지 않았습니다.**

학습/예측 데이터 중복, 다른 분리·튜닝·학습량·모델 설정 때문에 리더보드와 이번 OOF 점수는 직접 비교할 수 없습니다. 점수 차이 전체를 누수 하나의 효과로 설명하지 않습니다. 기존 전체 label 기반 Morgan8 특성 선택과 바깥 fold를 조기 종료에 사용한 기록은 독립 성능의 근거로 사용하지 않습니다.

## 로컬 실행

검증 환경은 macOS ARM64, Python 3.11.15이며 외부 API·클라우드·GPU를 사용하지 않습니다. Python 3.11 가상환경을 권장합니다. 원본 데이터는 포함하지 않으므로 이용 권한이 있는 CSV를 `data/`에 별도로 준비합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

macOS에서 LightGBM이 OpenMP 라이브러리를 찾지 못하면 별도 `libomp` 설치가 필요할 수 있습니다. 설치 명령은 자동 실행하지 않습니다. 노트북 열람에는 Jupyter/IPython 환경을 별도로 사용합니다.

학습 CSV: `SMILES`, `ecfp_0`~`ecfp_1023`, `fcfp_0`~`fcfp_1023`, `ptfp_0`~`ptfp_1023`, `MolWt`, `clogp`, `sa_score`, `qed`, `label`. 예측 CSV는 `label`만 제외합니다. 필수 열 누락·추가·중복, 비이진 Fingerprint, 비유한 수치, 잘못된 SMILES를 거부합니다. 결측치를 임의 보정하지 않습니다. SMILES만 받는 예측기는 아니며 **제공된 Fingerprint와 Descriptor의 생성 정의가 같은 입력**이 필요합니다.

학습 없이 데이터 집계만 확인하려면:

```bash
python -m toxicity.train audit \
  --data data/train.csv \
  --prediction-data data/predict_input.csv \
  --output artifacts/quality-v1.json
```

전체 평가·최종 학습·예측(출력은 새 폴더 사용):

```bash
python -m toxicity.train train \
  --data data/train.csv \
  --prediction-data data/predict_input.csv \
  --output artifacts/retrospective-v1
```

- `report.json`: 데이터 품질·환경·평가 설정·fold별 및 전체 집계
- `model.joblib`: 상수 제거와 LightGBM이 묶인 Pipeline 및 임계값·스키마·라벨 정의
- `provenance.json`: 로컬 입력 파일 SHA-256
- `predictions.csv`: 입력 SMILES 순서를 유지한 `SMILES,output` 두 열

저장 모델로 별도 프로세스에서 다시 예측:

```bash
python -m toxicity.train score \
  --data data/predict_input.csv \
  --model artifacts/retrospective-v1/model.joblib \
  --output artifacts/predictions-reloaded.csv
```

기존 결과·모델·예측 파일은 덮어쓰지 않습니다. joblib 로드는 코드를 실행할 수 있으므로 직접 만든 신뢰 가능한 모델만 사용합니다. 원본·모델·개별 예측·파일 해시는 `artifacts/`와 `data/`에 보관하고 Git에서 제외합니다.

## 검증과 파일 안내

자동 검사 **11개 통과**: 입력 스키마, 분자 표준화·입체 정보, 전이적 그룹 합치기, 라벨 충돌 보고, fold 중복/OOF coverage, 독성 지표 방향, 임계값 동점 정책, 학습 전처리 격리, 바깥 데이터 격리, 실제 LightGBM 내부 학습 및 모델 저장·로드·예측 검사를 포함합니다.

실제 교육 데이터 8,349행 전체 평가를 완료했습니다. 최종 저장·로드 확률 차이는 **0.0**, 예측 927행을 생성했습니다. 검증된 환경과 결과의 범위는 [검증 집계](reports/verification.json)에 기록합니다. AMD64·Windows 실행, 새 독립 데이터 성능, 배포·부하·의약 안전성은 검증하지 않았습니다.

- `01_eda.ipynb`, `02_modeling.ipynb`: 기존 수업·제출 실험 보관본. 과거 결과는 이번 재실행 결과가 아닙니다.
- `03_reproducible_evaluation.ipynb`: 공개 집계 결과를 원본 없이 읽는 노트북
- `toxicity/`: 입력 검증·분자 그룹·평가·학습·점수화 실행 코드
- `tests/`: 데이터 및 평가 경계 자동 검사

## 출처와 개인 개선 기록

기존 수업·팀 프로젝트를 기반으로 한 포트폴리오입니다. 초기 실험을 처음부터 독자 구현했다고 주장하지 않으며 이번 데이터·평가 개선은 커밋으로 구분합니다. 원저작물에 임의 라이선스를 부여하지 않았습니다. 교안·녹취록·원본·모델·DB·키·개별 예측·노트북 실행 출력은 커밋하지 않습니다.

설계 참고: [scikit-learn 중첩 교차검증](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [데이터 누수와 전처리](https://scikit-learn.org/stable/common_pitfalls.html), [RDKit Murcko scaffold](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html).
