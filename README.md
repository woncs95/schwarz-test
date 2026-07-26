# Politikbereich classification

무료 오픈소스 도구로 분류 모델을 탐색하고, 실험을 추적하며, 보고서를 생성하는 프로젝트입니다.

## 구조

```text
configs/       실험 설정
data/raw/      추가 원본 데이터(현재 index.csv는 루트에서 그대로 사용)
data/processed 생성 데이터
models/        학습 모델
notebooks/     Jupyter 탐색
reports/       Quarto 문서와 생성 결과
src/           재사용 가능한 학습 코드
tests/         테스트
```

## Windows 빠른 시작

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

베이스라인 최적화 및 MLflow 기록:

```powershell
python -m src.train --config configs/base.yaml
mlflow server --backend-store-uri ./mlruns --port 5000
```

학습이 끝나면 다음 결과가 생성됩니다.

- `models/best_model.joblib`: 최종 scikit-learn 파이프라인
- `reports/generated/evidently_classification.html`: Evidently 오류 분석
- `reports/generated/classification_report.csv`: 클래스별 지표
- `reports/generated/confusion_matrix.csv`: 혼동 행렬
- `mlruns/`: MLflow 실행, metric 및 모델 artifact

Quarto CLI를 설치한 뒤 보고서 생성:

```powershell
quarto render reports/report.qmd --to html
```

검증:

```powershell
pytest
ruff check src tests
```

Docker 사용:

```powershell
docker compose up --build
```

`index.csv`는 이미지에 포함하지 않으며 실행 시 볼륨을 통해 접근합니다. `.env`도 Git 및 Docker 이미지에서 제외됩니다.
