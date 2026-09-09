# 재현 가능 범위

Python 3.10 이상을 사용한다. 기본 집계 도구는 표준 라이브러리만 필요하다. 구성 검사와 전체 테스트에는 [requirements-audit.txt](../requirements-audit.txt)의 PyYAML 6.0.3이 필요하다. 이 의존성은 2026년 검증용이며 원 GPU 학습 환경의 lock이 아니다.

```bash
python scripts/verify_historical_results.py --output docs/validation/offline-verification.json
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v
python scripts/check_training_config.py historical/git-20251214/configs/exp_stage3_full_warmstart.yaml
python scripts/check_run_manifest.py experiments/historical-stage1.json
```

현재 첫 명령은 통과하고, 테스트 19개도 통과한다. 마지막 두 명령은 구성 불일치와 기록 누락을 발견하여 종료 코드 1을 반환한다. 구성 검사는 Stage3에서 무시되는 필드와 학습 단계가 0개인 문제를 찾는다. 실행 기록 검사는 모델 revision, split, checkpoint, GPU, 시간 등의 공백을 드러낸다. 형식상 기록이 완성되어도 실행 진위를 증명하지는 않는다.

| 자료 | 복구·검증 상태 |
|---|---|
| 학습·전처리·평가 코드 | 두 snapshot으로 복구, Python 문법 검사 통과 |
| 모델 설정 | Stage3 Mistral v0.3 확인, Stage1/2 기본 Llama와 평가 라벨 불일치 |
| 실제 모델 revision·checkpoint | 미확보 |
| 데이터 ID·샘플 상한 | 복구; 실제 로드 성공 여부와 최종 행 수는 미확인 |
| 독립 분할·중복 제거·학습 loss·시간 | 미확보 |
| 원 requirements | 복구; 전체 버전 lock 없음 |
| 개별 평가 점수 | Git의 5개 파일, 14개 그룹 평균 검증; 반복 key는 경고 |
| 최종 자동 평가 summary | 제출 파일 4개 복구; 대응하는 생성 원문 전체는 미확보 |
| judge 판정 | Git의 익명화 결정 파일 8개 집계 검증; 제출 summary 5개의 원시 판정은 미확보 |
| GPU·외부 API 재실행 | 하지 않음 |

`experiments/historical-stage1.json`은 **실제 평가 실행의 불완전한 기록**이다. 복구한 기본 config가 그 실행과 일치한다고 확정할 수 없어 모델 ID 등을 임의로 채우지 않았다. 정적 설정값은 [fine-tuning.md](fine-tuning.md)에 별도로 기록한다.

## 과거 명령을 그대로 실행하지 않는 이유

원 trainer는 모델을 먼저 로드한 뒤 활성 stage를 순회한다. 비호환 Stage3 구성을 그대로 실행하면 큰 모델부터 다운로드하고도 학습하지 않을 수 있다. 먼저 preflight를 확인하고 정합한 원 runner와 adapter를 복구해야 한다. 원 의존성을 근거 없이 최신 버전으로 바꾸거나 새 trainer를 원 구현처럼 만들지 않았다.

실제 재학습에는 데이터 source·subset·license·revision 검증, 원 분할 복구 또는 명시적인 새 분할 설계, config와 실제 실행의 연결, GPU와 정확한 환경, checkpoint 계보가 필요하다. judge 재실행은 외부 API 비용과 의료 텍스트 전송을 수반하므로 이번 작업에서는 수행하지 않았다.

새 집계 도구는 저장된 평균·분모·승자 수를 검사한다. 원문으로 ROUGE/BERTScore를 다시 계산하거나 API 응답을 재판정하지 않는다. corpus BLEU를 sentence BLEU의 평균으로 대체하지 않는다. 반복 key를 조용히 제거하지 않고 경고하여 역사적 결과의 가중치를 보존한다.
