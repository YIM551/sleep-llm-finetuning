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
