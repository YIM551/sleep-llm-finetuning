# 데이터 출처와 전처리

아래 목록은 [원 prepare_datasets.py](../historical/git-20251214/src/prepare_datasets.py)의 **로드 대상 설정**이다. 성공한 dataset 과 실제 학습 행 수를 나타내지 않는다. source 별 예외를 잡아 skip 하고 실제 다운로드/전처리 로그는 미확보다. 2026-09-10 공식 HF card/API 의 현재정보를 확인했으며 현재 revision 을 과거 학습 revision 으로 사용하지 않았다.

| 단계 | 지정 ID / 공식 출처 | 코드 상한 | 현재 라이선스·접근 확인 |
|---|---|---:|---|
|1|[Malikeh1375/medical-question-answering-datasets](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets)|50,000|MIT 표기;12 개 subset 인데 코드 subset 지정 없음|
|1|[lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD)|20,000|card license metadata 없음, 원권리 확인 필요|
|1|SleepQA|5,000|namespace 없는 식별자, 현재 API401; 정확한 source 확인 필요|
|1|[bigbio/mediqa_qa](https://huggingface.co/datasets/bigbio/mediqa_qa)|5,000|unknown 표기|
|1|[truehealth/medicationqa](https://huggingface.co/datasets/truehealth/medicationqa)|없음|card license metadata 없음|
|2|[avaliev/chat_doctor](https://huggingface.co/datasets/avaliev/chat_doctor)|60,000|Apache2.0 표기|
|2|[UCSD26/medical_dialog](https://huggingface.co/datasets/UCSD26/medical_dialog)|40,000|unknown 표기|
|2|[Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)|20,000|현재 gated; card RAIL-D/API other, 변형·재배포 조건 있음|

## 공통 포맷과 처리

```json
{"instruction":"<task instruction>","input":"<question>","output":"<reference answer>"}
```

위는 필드설명용 가상 구조다. 실제 의료기록 예시를 재배포하지 않는다. QA 는 question/input/prompt 등의 alias 와 answer/output 등을 매핑한다. 상담은 dialogue 의 마지막 patient/doctor, 문자열 conversation 등을 매핑한 뒤 QA 방식으로 fallback 한다. seed42 로 source 별 제한과 최종 shuffle 을 수행하고 JSONL 로 저장한다. feature engineering 이나 scaler/tabular target 분류를 수행한 프로젝트로 꾸미지 않는다. 이 프로젝트의 모델링대상은 텍스트 생성이며 별도 EDA notebook 은 발견하지 못했다.

원 코드의 `Amod` mapper 에는 공식 schema 의 대문자 Context/Response alias 가 없다. 다중 config 데이터의 subset 선택, 지원하지 않는 record 처리, 데이터별 버전변화 때문에 새환경에서 모든 source 를 다시 받을 수 있다고 단정할 수 없다. 학습용 factual80,000/counsel120,000 은 config 의 **상한**이다.

## Split / leakage / 평가 표본

전처리에서 독립 train/validation/test 분할이나 환자·대화 중복 제거는 찾지 못했다. 원 trainer 는 eval_dataset 을 받지 않고, 원 eval_generation 기본입력은 학습 JSONL 의 첫 100 행이다. 기본명령 그대로라면 학습 데이터를 평가에 재사용할 위험이 있다. 보고서의 별도 eval_fixed200 이 실제로 held-out 인지 입증할 split hash 는 없다.

저장된 자동평가 summary 의 유효/전체는 Stage1 185/200, Stage2 188/200, Stage3 180/200 이다. 원 collect_eval_metrics 는 빈/잘못된 레코드를 건너뛰므로 제외이유별 분모는 복구된 원 답변 없이는 완전히 재구성할 수 없다. 일부 per-example 파일의 Stage1 에는 동일 question/reference key3 개 반복, 초기 Stage3 에는 1 개 반복이 있다. [원시스코어 검증](validation/offline-verification.json)은 기존 가중치를 변경하지 않고 반복을 경고한다.

## 공개·다운로드 범위

학습 원문·weights 는 복구되지 않았고, 의료 QA 출력에는 연락처와 제 3 자 문장이 포함되어 원문을 제외했다. 다운로드를 자동실행하지 않는다. 정확한 source/subset/revision/권리를 확인한 후 정합한 전처리와 split manifest 를 별도새 실험으로 구성해야 한다. URL 의 현재데이터를 받아 과거 실험과 같다고 표시해서는 안 된다.
