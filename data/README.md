# 결과 전사와 복구 결과의 구분

`reported-evaluation.csv`는 2025년 원 보고서 인쇄 쪽 7~12의 **역사적 전사본**이다. 이번에 실험을 실행해서 생성한 CSV가 아니다. 자동 점수의 정밀값과 분모는 [제출 summary](../historical/submission-20251215/reports/)에서 확인한다.

전사본의 judge Stage2 0.74, Stage3 0.90/0.92에는 Reference를 외부 LLM에 노출한 refined2pass 후처리가 있다. Stage1 0.66(50건)은 복구한 별도 summary의 110/186과 다른 실행이다. 전사 수치를 바꾸어 과거 보고서를 덮어쓰지 않고 [정정과 전체 실험표](../docs/experiments.md)에서 구분한다. 이 전사본을 순수 FT 성능의 근거로 그대로 사용하면 안 된다.

학습 데이터와 개인 의료 원문은 포함하지 않는다. BLEU와 ROUGE/BERTScore의 척도를 혼합하지 않는다.
