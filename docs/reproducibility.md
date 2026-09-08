# 재현에 필요한 원본

| 항목 | 현재 상태 |
| --- | --- |
| 정확한 base 모델 repo/revision/tokenizer | 확인 필요 |
| Stage별 데이터 출처·라이선스·샘플 수 | 확인 필요 |
| train/validation/test split 및 중복 방지 | 확인 필요 |
| 학습 스크립트와 dependency lock | 미확보 |
| LoRA 여부·학습률·epoch·batch·seed·하드웨어 | 확인 필요 |
| stage checkpoint와 warmstart 연결 | 미확보 |
| 자동지표 라이브러리·토큰화·집계 방식 | 확인 필요 |
| judge system/user prompt·답변 순서·invalid 처리 | 확인 필요 |
| 원시 생성 답변과 평가 JSON | 미확보 |

현재 결과를 재현했다고 주장하지 않으며, 체크포인트나 코드를 새로 작성하여 과거 구현처럼 표시하지 않습니다.

## 누락 항목을 실행으로 확인하기

2026-09-09에 Python 표준 라이브러리만 사용하는 `scripts/check_run_manifest.py`를 추가했습니다. 학습 스크립트가 아니라 실험 기록의 완성도 검사입니다. Python 3.10 이상에서 설치 없이 실행할 수 있습니다.

```bash
python scripts/check_run_manifest.py experiments/historical-stage1.json
python -m unittest discover -s tests -v
```

첫 번째 명령은 **종료 코드 1**과 `metadata_complete: false`가 정상적인 현재 결과입니다. 보고서에 없는 값을 `null`로 유지했기 때문입니다. 두 번째 명령은 합성 메타데이터로 검사기 자체를 검증하며 모델을 학습하거나 평가하지 않습니다. 반환 값 0은 메타데이터 검사 통과, 1은 누락·형식 오류·내부 모순입니다.

`experiments/historical-stage1.json`은 보고서 Stage1을 추적하기 위해 새로 만든 불완전한 기록입니다. `reported_model_label`을 실제 모델 ID로 대입하지 마세요. 원본에서 확인한 값만 채우고, Stage2/Stage3는 독립 run ID와 별도 기록으로 남겨야 합니다. Stage3 warmstart에는 부모 Stage2 checkpoint SHA-256도 요구됩니다.

검사 항목:

- 모델·tokenizer의 정확한 ID와 40자리 commit revision, 데이터 snapshot SHA-256
- 전처리 후 총 예제 수와 train/validation/test 수의 합, split ID 목록의 fingerprint
- 데이터 출처·라이선스·분할/중복 방지 방법, seed·optimizer·학습률·epoch·batch·sequence length
- 튜닝 방식과 base precision, LoRA/QLoRA이면 실제 target modules/rank/alpha/dropout
- GPU/VRAM·측정 학습 시간, 코드·환경 lock·checkpoint·평가 출력의 SHA-256

검사기는 hash 형식·필수 값·수의 합 등을 확인합니다. 원본 파일 존재 여부, hash의 진위, split의 실제 교집합, 데이터 누수, 라이선스 적합성, 실행 성공이나 결과 재현은 확인하지 않습니다. 같은 split fingerprint를 발견하면 실패하지만 서로 다른 fingerprint만으로 중복이 없다고 증명할 수는 없습니다. 실제 artifact와 원시 데이터의 별도 감사가 필요합니다. GPU가 필요 없는 실험은 현재 이 LLM 학습 기록 형식의 대상이 아닙니다.

이 검사기에서 QLoRA는 원 논문의 고정 4비트 base + LoRA를 뜻합니다. 다른 양자화 학습 방식은 이름과 설정을 분리하여 기록해야 합니다. GPU 이름만 보고 method 값을 추정하지 마세요. [기술 설명과 확인 범위](technical-evidence.md)를 함께 확인하세요.
