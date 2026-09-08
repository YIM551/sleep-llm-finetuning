# 모델·학습 방식의 확인 범위

2026-09-09에 보고서, 공개 저장소 전체 파일과 Git 이력을 다시 대조했습니다. 아래 기술 설명은 원 논문·공식 문서를 확인한 개념 정리이며, 과거 학습 설정을 복원한 결과가 아닙니다.

## 무엇을 확인했는가

| 항목 | 확인된 근거 | 남아 있는 공백 |
| --- | --- | --- |
| 모델 | 원 보고서 p.6의 `base_mistral7b` | 정확한 repo ID, Base/Instruct, v0.x, commit revision, tokenizer |
| 학습 설계 | p.6의 factual/counseling/full, Stage3의 Stage2 warmstart와 base scratch | 학습 코드, 체크포인트 연결, 대상 파라미터 |
| 튜닝 방식 | p.6은 LoRA 여부·학습 하이퍼파라미터를 명시하지 않음 | LoRA/QLoRA/full 여부, target modules, rank, dtype, 양자화 |
| 데이터 | pp.5–6의 수면 지식형·상담형·통합형 분포 설명 | dataset ID/revision/license, 원본·전처리 후 수, split, 중복·오염 검사 |
| 환경 | 보고서에 실행 환경 상세 없음 | GPU 모델·VRAM, 학습 시간, dependency lock, seed |
| 평가 | pp.7–12의 집계 지표, base 대 FT, judge `gpt-4o-mini` | 원시 답변, 평가 코드·프롬프트, 지표 구현 버전, 샘플 선택 |

공개 저장소 최초 두 커밋은 README 생성과 보고서 정리입니다. 과거 학습 코드가 포함된 Git 이력은 없습니다. `src`, 학습 notebook, requirements/lock, API, Docker, DB schema가 없는 것은 이 공개본이 결과 기록이라는 범위와 일치합니다. 학습 구현이 검증되었다는 뜻은 아닙니다.

## Mistral 7B라는 이름만으로 부족한 이유

7B는 파라미터 규모를 나타내며 정확한 체크포인트를 지정하지 않습니다. 공식 `Mistral-7B-v0.3` 모델 카드도 v0.2와의 어휘 차이를 명시합니다. 모델과 tokenizer의 저장소 ID, Base/Instruct 구분, 불변 commit revision을 함께 기록해야 합니다. 이 예시는 원 실험이 v0.3이었다는 근거가 아닙니다. [Mistral 공식 모델 카드](https://huggingface.co/mistralai/Mistral-7B-v0.3)

## LoRA·QLoRA·Unsloth 구분

- **LoRA**: 사전 학습 가중치를 고정하고 선택한 가중치 행렬의 변화량을 저랭크 행렬로 학습합니다. 일반적인 표기는 `W' = W + scale × B × A`입니다. 출력층 하나를 끝에 덧붙이는 방식으로 정의하지 않습니다. 실제 적용 위치는 학습 코드의 target modules로 확인해야 합니다. [LoRA 원 논문](https://arxiv.org/abs/2106.09685)
- **QLoRA**: 원 논문은 고정된 4비트 양자화 base를 통해 LoRA adapter로 gradient를 전달하고 NF4, double quantization, paged optimizer를 설명합니다. 모든 연산과 학습 파라미터가 4비트라는 뜻은 아닙니다. [QLoRA 원 논문](https://arxiv.org/abs/2305.14314)
- **Unsloth**: LoRA/QLoRA 등을 지원하는 학습 도구입니다. LoRA 기법 자체와 같은 이름으로 취급하지 않습니다. 이 저장소에서 사용했다는 근거는 없습니다. [Unsloth 공식 저장소](https://github.com/unslothai/unsloth)

어떤 GPU를 썼다는 기억만으로 양자화가 필요했는지 판단할 수 없습니다. base precision, sequence length, batch, gradient accumulation, optimizer, activation/checkpointing 설정과 실제 peak memory를 확인해야 합니다. 위 설명은 앞으로 확인할 판단 기준입니다. 본 실험의 A100 사용 여부와 양자화 선택 이유는 미확인입니다.

## 데이터와 결과를 설명하는 기준

데이터 양은 모델 크기 하나로 충분·불충분을 단정할 수 없습니다. 원시 행 수, 필터 후 대화 수, token 수, 분포, 중복, 학습·검증·시험 분할과 held-out 결과를 함께 기록해야 합니다. 이 프로젝트에는 특정 최소 샘플 수의 충분성을 검증한 실험이 없습니다.

결과 비교 대상은 보고서의 `base_mistral7b`입니다. 다른 상용 모델이나 GPT-2와의 동일 조건 비교는 없습니다. 보고서의 base 대비 개선을 인정하되, 그것이 다른 모델보다 우수·열등하다는 결론으로 이어지지 않습니다. Stage 간 데이터 분포가 다르므로 stage별 점수 변화에서 누적 학습의 인과 효과도 분리할 수 없습니다.

Stage3 scratch의 명칭에 나온 `Full FT`는 통합 분포 학습을 가리키는 맥락이며, 모든 파라미터를 업데이트하는 full-parameter fine-tuning이었다고 확정할 수 없습니다. 보고서 p.6은 LoRA 여부를 생략했습니다. 또한 여기서 scratch는 사전 학습된 base에서 시작한다는 뜻이며 무작위 가중치부터 언어모델을 사전 학습했다는 뜻이 아닙니다.

## 다음 검증 순서

1. 학습 스크립트·환경 lock·model/tokenizer ID/revision·adapter config·GPU 로그를 원본에서 복구합니다.
2. 데이터의 출처·라이선스와 전처리 후 분할 ID를 고정하고, 환자/대화/문서 단위 중복과 test 오염을 확인합니다.
3. 같은 held-out 질문과 generation 설정으로 base/FT 원시 답변을 저장하고, 지표·judge prompt·답변 순서를 고정해 다시 평가합니다.
4. 그 후에만 모델 교체나 LoRA/QLoRA 선택을 비교합니다. 새 모델이 더 좋다는 결과와 재학습 성공 여부는 현재 **측정되지 않음**입니다.

모든 단계에서 과거 기록 복구와 신규 실험을 별도 run ID로 관리합니다. [메타데이터 검사 도구](reproducibility.md)는 누락을 드러내기 위한 도구이며 학습·성능·의료 안전성 검증을 대신하지 않습니다.
