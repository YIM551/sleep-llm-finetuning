# 기술 주장과 실제 근거

2026-09-10 과거 Git ref 와 제출코드를 복구했다. 이전의 '학습/평가 코드 없음, 모델버전 전부 미확인' 판단은 정정한다. 실제 실행 메타데이터까지 복원된 것은 아니다.

| 주장 | 확인 범위 | 근거 |
|---|---|---|
| 모델 | Stage3 config Mistral7B Instruct v0.3, Stage1/2 기본 Llama3 8B Instruct | [학습 설정](fine-tuning.md) |
| QLoRA | NF4 4bit+rank64 LoRA 구현 존재 | [원 trainer](../historical/git-20251214/src/train_qlora.py) |
| 데이터 |8 개 로드대상 ID, 성공 source/최종행수 미확인|[dataset](dataset.md)|
| 평가 | 저장점수 14cohort 평균·judge8 파일 집계 일치 | [검증](validation/offline-verification.json) |
| 높은 선호율 | 정답을 외부 LLM 에 제공한 refined2pass 후보 포함 | [실험 교란](experiments.md) |
| A100 사용 | 원 README 목표환경 예시, 실 GPU 로그 없음 | [환경](fine-tuning.md) |
| 서비스 연결 | adapter 를 실제 RAG endpoint 에 연결한 증거 미확보 | [architecture](architecture.md) |

점수 개선은 의료 정확성·실제 수면개선·상용동시성·다른 상용모델 대비 우월성을 입증하지 않는다. [provenance](provenance.json)에서 원본과 신규검증을 구분한다.
