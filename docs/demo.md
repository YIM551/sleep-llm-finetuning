# 실행 결과 자료

모델 학습·평가 실험이므로 대표증거는 실제 저장된 점수와 figure 다. 채팅 UI/모델추론 시연영상은 복구하지 못했다. 그림 40 개를 시각검토했고 얼굴·실명·연락처가 없어 보존했다. [원본 hash/경로](provenance.json)로 추적할 수 있다.

| 자료 | 무엇을 보여주는가 | 읽을 때 주의 |
|---|---|---|
|[대표 ROUGE 비교](../historical/git-20251214/reports/eval_compare_stage2set_det_fair_v4/rouge_mean_bar.png)|동일 Stage2set 의 base/Stage2/Stage3 자동점수|저장된점수, 실제실행 model 연결과 held-out 여부 미확인|
|[BERTScore 분포](../historical/git-20251214/reports/eval_compare_stage2set_det_fair_v4/bertscore_f1_box.png)|평균뿐 아니라 개별점수 분산|임상안전성 검사가 아님|
|[답변 길이 분포](../historical/git-20251214/reports/eval_compare_stage2set_det_fair_v4/pred_chars_box.png)|후보간응답길이 차이|길이와품질을 동일시하지 않음|
|[원 비교 그림들](../historical/git-20251214/reports/)|초기/Stage 별/정제조건별 30 개그림|서로 다른 cohort 를 한실험으로 합치지 않음|
|[제출그림 10 개](../historical/submission-20251215/reports/plots_finetune_better/)|개선된지표와 우세한후처리후보를 선택한그림|[selection/refined2pass 교란](experiments.md) 필독|
|[익명화 원 보고서](reports/course-report-anonymized.pdf)|당시 실험목적·방법·해석|표지생략, 인쇄쪽 2~17 보존. 해석은 현재정정문서 우선|
|[오프라인 검증 JSON](validation/offline-verification.json)|원수치재집계와 hash 확인|2026 신규검증, 원학습/추론로그가 아님|

README 의 이미지를 본 뒤 [실험별판정표](experiments.md), 해당 historical CSV, [재집계 도구](../scripts/verify_historical_results.py) 순으로 확인하면 실행 없이도 근거를 검토할 수 있다. 개인정보/제 3 자의료문장이 들어간 raw pairwise 답변은 공개하지 않고 키·라벨·승자만 decisions.csv 에 남겼다.
