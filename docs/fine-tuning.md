# 모델과 학습 설정의 근거

2026-09-10 원 Git ref 와 제출 코드를 복구했다. 구성값과 실제 실행 증거를 구분한다. 이전 공개본의 '학습 코드 미확보' 판단은 정정한다.

| 단계/출처 | 모델 식별자 | 확인 범위 |
|---|---|---|
| factual/counsel/curriculum 기본 | `meta-llama/Meta-Llama-3-8B-Instruct` | [base.yaml](../historical/git-20251214/configs/base.yaml); 평가 라벨 base_mistral7b 와 다름 |
| Stage3 scratch | `mistralai/Mistral-7B-Instruct-v0.3` | [scratch YAML](../historical/git-20251214/configs/exp_stage3_full_scratch.yaml), commit54708b3 |
| Stage3 warmstart | `mistralai/Mistral-7B-Instruct-v0.3` | [warmstart YAML](../historical/git-20251214/configs/exp_stage3_full_warmstart.yaml) |
| 실제 adapter base/revision | 확인 필요 | adapter_config/weights 미확보 |

Mistral 공식 모델은 7B 계열이며 Instruct 는 base 를 지시 학습한 버전이다. 질의응답 목적과 맞지만 당시 Base/Instruct 비교실험으로 선택했다는 기록은 없다. [공식 모델 카드](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

| 학습 항목 | 원 trainer/config 값 |
|---|---|
| 양자화 | NF4 4bit, double quant false, compute bfloat16 |
| LoRA | rank64, alpha16, dropout0.05, bias none |
| target_modules | q/k/v/o_proj, gate/up/down_proj |
| batch / gradient accumulation | 2 /16; 장치 1 개라면 effective32 설정 |
| learning rate / scheduler / warmup | 2e-4 / cosine /0.03 |
| epoch / sample cap | factual2 /80,000; counsel1 /120,000; 실제행수 아님 |
| seed / gradient checkpointing / bf16 | 42 /true /true |
| sequence length | tokenizer.model_max_length; Stage3 YAML1024 는 연결 안 됨 |
| optimizer | 명시하지 않음; 당시 버전 기본값 확정 불가 |
| validation | train_stage 에 eval_dataset 없음; loss/validation log 미확보 |

근거: [train_qlora.py](../historical/git-20251214/src/train_qlora.py), [utils.py](../historical/git-20251214/src/utils.py), [configs](../historical/git-20251214/configs/).

LoRA 는 출력층 하나를 추가하는 방식이 아니라 선택한 행렬의 저랭크 변화량을 학습한다. 이 원본에는 고정 4bit base+LoRA 를 결합한 QLoRA 계열 구현이 있다. QLoRA 논문의 모든 기법을 쓴 것은 아니다(double quant=false, paged optimizer 지정 없음). Unsloth 사용 근거는 없다. [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314)

## 바로 재학습할 수 없는 이유

- Stage3 에 enabled/file 이 없어 원 trainer 의 학습 단계 0 개. train_file 을 읽지 않는다.
- init_adapter_dir/resume_from_checkpoint 를 쓰는 코드가 없어 warmstart 성공을 config 만으로 증명할 수 없다. 중첩 train/lora/max_length 도 사용하지 않는다.
- base 의 unquoted `evaluation_strategy: no`는 PyYAML6 에서 boolean False 다. trainer 의 문자열 전략과 다르다.
- model/data revision, checkpoint, 최종행수/분할 hash 가 없다. dataset download 성공도 확정할 수 없다.

원본을 바꿔 과거 성공을 재구성하지 않았다. [preflight](../scripts/check_training_config.py)는 모델을 import 하지 않고 이 문제를 보고한다. `scratch`는 사전학습 base 에서 시작한다는 뜻이며 `full`이라는 이름으로 전체 파라미터 학습을 단정할 수 없다.

## 환경과 미확인 숫자

원 README 의 A100 MIG3g.40GB 는 목표환경 예시이며 실 GPU 로그가 아니다. 부분 백업에서 Python3.10 경로, PEFT0.18.0, Accelerate1.12.0, CUDA-runtime12.8.90 설치 metadata 를 찾았으나 해당학습 run 의 lock 으로 확정할 수 없다. GPU/VRAM/RAM, 실제 PyTorch/Transformers/CUDA 조합, 학습시간, peak memory, loss, checkpoint hash 는 **확인 필요**다. [실행 기록 검사](reproducibility.md)는 이 공백을 유지한다.
