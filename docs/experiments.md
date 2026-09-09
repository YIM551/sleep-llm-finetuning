# 실험 결과와 해석 정정

원 실험자료는 2025-12-12~15 에 만들어졌다. 2026-09-10 에는 자료를 복구하고 **저장점수 재집계**만 수행했다. 학습·추론·judge API 를 재실행하지 않았다. 과거 보고서의 성능주장을 원 코드와 대조하여 다음과 같이 구분한다.

## Goal → Setup

의료/수면 상담 답변을 지식형·상담형·통합데이터로 학습하고 문자열유사도와 judge 선호를 비교하려 했다. factual-only, counsel-only, curriculum(두 단계), Stage3 scratch/warmstart 구성은 서로 구분한다. [정확한 config 와 실행불일치](fine-tuning.md), [데이터 설정](dataset.md)을 참조한다. 체크포인트나 고정 modelrevision 이 없어 configuration 이 실제 run 을 증명하지 않는다.

## 자동 지표

원 도구는 ROUGE(use_stemmer=true, per-example), sacrebleu 의 sentence/corpus BLEU, BERTScore F1 을 계산한다. 제출 summary 에서 scorer 는 distilbert-base-uncased, device=cuda 이며 collect_eval_metrics 는 rescale_with_baseline=false 다. BERTScore 의 cuda 는 평가장치 기록이지 학습 GPU 모델을 뜻하지 않는다.

| 제출 최종 summary | used/total | ROUGE-1 base→후보 | BERT F1 base→후보 | corpus BLEU base→후보 |
|---|---:|---|---|---|
|Stage1|185/200|0.3159→0.3939|0.7860→0.8199|3.9394→17.1373|
|Stage2|188/200|0.2701→0.3161|0.7609→0.7856|1.2334→8.0054|
|Stage3 warmstart|180/200|0.2740→0.2985|0.7634→0.7664|1.4617→7.5495|
|Stage3 scratch/full|180/200|0.2740→0.3041|0.7634→0.7648|1.4617→8.1795|

근거: [제출 CSV4 개](../historical/submission-20251215/reports/). 보고서값과 일치하지만 전체 생성 JSONL 과 split 정보가 없어 임상 정확성/held-out 일반화를 단정하지 않는다. Git 에서 복구한 초기/정제 비교 5 개 per-example 파일 14 개 cohort 의 산술평균은 모두 해당 summary 와 일치했다. **제출 최종 CSV4 개 전체의 원시 생성을 다시계산했다는 뜻은 아니다.** corpus BLEU 는 sentence BLEU 평균으로 복원할 수 없다.

## 초기 실패와 후속 비교를 모두 보존

| Git 결과 파일 그룹 | 판정 수 | 저장된 판정 |
|---|---:|---|
|pairwise/base_vs_stage2|188|base188, Stage2 0|
|pairwise/base_vs_stage3_on_stage2|188|base188, Stage3 0|
|pairwise/stage1_vs_stage3_on_stage1|185|Stage1 87, Stage3 68, tie30|
|pairwise/stage2_vs_stage3|188|Stage2 77, Stage3 100, tie11|
|pairwise_llm/base_vs_stage2|50|base50, Stage2 0|
|pairwise_llm/base_vs_stage3_on_stage2|50|base50, Stage3 0|
|pairwise_llm/stage2_vs_stage3_on_stage2|188|Stage2 81, Stage3 93, tie14|
|pairwise_llm_det/stage2_det_vs_stage3_det_cleaned_v4|188|Stage2 91, Stage3 57, tie40|

원 summary 와 개인정보를 제거한 decisions.csv 는 [historical/git-20251214/reports](../historical/git-20251214/reports/)에 있다. 8 개 파일의 승자별계수와 분모를 재집계해 summary 와 일치함을 확인했다. evaluator 라벨은 gpt-4o-mini 다. 모델 고정 revision·원 API 요청은 없어 공급자 수준 실행까지 독립 검증한 것은 아니다.

초기 실험/정제/다른 rubric/후처리를 섞어 '단계가 진행될수록 향상'이라 말할 수 없다. 낮은 품질의 원인으로 promptformat, 데이터정합성, 학습설정, 평가기준 불일치가 가능하지만 각각을 통제한 ablation 이 없어 원인을 확정하지 않는다.

## 높은 선호율의 교란

[refine_2pass.py](../historical/submission-20251215/tools/refine_2pass.py)는 질문·**Reference**·원답변을 GPT-4o-mini 에 보내 1 차로 재작성하고, Reference 와 draft 로 2 차검토한 뒤 generated 를 덮어쓴다. 즉 FT 답변 외에도 정답정보와 상용 LLM 이 추가로 들어간다.

| 제출 judge summary | 후처리 | 결과 |
|---|---|---|
|Stage1 pairwise.csv.summary|라벨상 refined2pass 아님|FT110/186=59.14%, base62, tie1, invalid13|
|Stage2 counsel|refined2pass|FT37/50=74%, base13|
|Stage2 overall|refined2pass|FT30/50=60%, base20|
|Stage3 warmstart faithful|refined2pass|FT45/50=90%, base5|
|Stage3 full faithful|refined2pass|FT46/50=92%, base4|

근거: [제출 pairwise_llm_ref](../historical/submission-20251215/reports/pairwise_llm_ref/). 원 보고서 Stage1 66%(50 건/invalid1)는 별도실행이며 해당원시 판정이 없어 186 건 결과와 합치지 않는다. Stage3 90%와 92%는 전략끼리의 직접대결도 아니다. 정답노출/후처리 때문에 순수 FT 성능·의료안전성·일반화 개선으로 사용하지 않는다.

원 [plot_finetune_better_both.py](../historical/submission-20251215/tools/plot_finetune_better_both.py)는 개선한지표/우세한 FT 비교만 골라 그림을 만든다. 이 그림을 전체 실험의 대표값으로 제시하면 selection bias 가 생긴다. 위 실패표와 전체 summary 를 함께 공개했다. 제출 judge 는 rubric 별 기준이 다르고, 고정 A/B 위치이며 Git 초기 judge 의 위치 randomization 과도 다르다. resume 시 현재 batch 만 summary 하는 원 코드의 특성도 새 재실행에서 보완해야 한다.

## What I Learned / Next Experiment

좋은 지표와 학습의 인과효과는 다르다. 먼저 config→checkpoint→rawoutput→scorer 의 연결과 평가분모를 보존해야 한다. 다음실험은 동일 held-out 질문, 고정 modelrevision, 정답 노출 없는 생성, 양쪽동일 후처리, 답변순서 교환과 고정 rubric, 다중 judge/블라인드사람평가로 설계한다. 임상안전성이나 실제 수면행동변화는 별도 연구 대상이다.

## 이번에 실제 실행한 것

- 원 스코어 14cohort 평균 및 judge8 파일 집계: 통과.
- 복구 자료 122 개 published SHA256: 일치.
- metadata/config/집계 도구테스트 19 개: 통과.
- Stage3 preflight: 비호환 검출, 의도한 종료 코드 1.
- 불완전한 실제 run manifest: 누락검출, 의도한 종료 코드 1.
- **미실행:** GPU 학습/추론/새 judge 호출/새 benchmark/부하시험. 이 프로젝트에는 서비스 endpoint 가 없으며 RAG 부하 결과를 이 모델결과로 전용하지 않는다.

[기계 판독 검증보고서](validation/offline-verification.json), [재현명령](reproducibility.md).
