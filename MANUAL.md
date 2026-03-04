# Manual

## Environment
- OS: Ubuntu 22.04 (WSL2 포함)
- Python: 3.10.x

## Setup
```bash
cd ~/osl_project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Commands

### 1. End-to-End (Train + Eval + Plot)
```bash
python3 main.py --agent-type rsac --env-id OdorHold-v4 --total-episodes 20000
```

### 2. Train Only
```bash
python3 train.py --agent-type rsac --env-id OdorHold-v4 --total-episodes 20000 --out-dir runs
```

### 2-1. Train (Connectome Actor, `seq_len` 낮춰서 시도)
```bash
python3 train.py \
  --agent-type rsac \
  --env-id OdorHold-v4 \
  --total-episodes 20000 \
  --rsac-actor-backbone connectome2 \
  --connectome-hidden 180 \
  --connectome-steps 4 \
  --seq-len 6
```

### 3. Eval Only
```bash
python3 eval.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --episodes 50 --no-save-gif
```

### 4. Replot
- `train` 대상: 저장된 `plot_data`에서 학습 그래프 재생성
- `eval` 대상: 체크포인트로 다시 롤아웃해 trajectory 재생성

```bash
python3 replot.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --target all
python3 replot.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --target train
python3 replot.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --target eval --episodes 50
```

## Main Arguments (`main.py`)
- `--env-id`: default `OdorHold-v4`
- `--agent-type`: `drqn | dqn | rsac` (default `rsac`)
- `--total-episodes`: default `20000`
- `--seed`: default `42`
- `--force-cpu`
- `--reward-mode`: `mechanical | bio` (default `bio`)
- `--bio-reward-scale`: default `0.5` (`reward-mode=bio`일 때 concentration shaping 계수)
- `--cast-penalty`: default `0.025`
- `--turn-penalty`: default `0.01`
- `--b-hold`: default `0.5` (goal 영역 유지 보상)
- `--goal-hold-steps`: default `20`
- `--terminate-on-hold` / `--no-terminate-on-hold` (default terminate)

### RSAC-related
- `--lr-actor`: default `3e-4`
- `--lr-critic`: default `3e-4`
- `--lr-alpha`: default `3e-4`
- `--gamma`: default `0.99`
- `--tau`: default `0.005`
- `--rnn-hidden`: default `147`
- `--rsac-actor-backbone`: `gru | connectome | connectome2` (default `gru`)
- `--connectome-hidden`: default `180` (`connectome2`는 `90`의 배수이면서 `>= 180` 필요)
- `--connectome-steps`: default `4`
- `--batch-size`: default `128`
- `--seq-len`: default `16` (`connectome2` 실험에서는 `--seq-len 6`부터 확인 권장)
- `--buffer-size`: default `150000`
- `--learning-starts`: default `5000`

### Eval
- `--eval-episodes`: default `100`
- `--seed-base`: default `20000`
- `--rsac-actor-backbone`: `gru | connectome | connectome2` (기본은 run config 사용, 지정 시 override)
- `--connectome-steps`: 지정 시 run config override
- `--connectome-hidden`: 지정 시 run config override
- `--save-gif` / `--no-save-gif` (default save)
- `--plot-milestones` / `--no-plot-milestones` (default plot)

## Notes
- `RSAC`에서는 `--eps-start/--eps-end/--eps-decay-steps`가 실질적으로 사용되지 않습니다.
- 현재 connectome 계열 실험은 `connectome2` 사용을 권장합니다. 기존 `connectome`은 코드에 남아 있지만 주 실험 대상으로는 사용하지 않습니다.
- `OdorHold-v3`/`OdorHold-v4`의 spawn sampler는 balanced 방식으로 고정되어 있습니다.
- `OdorHold-v4`의 `reward-mode=mechanical`은 전통 RL shaping(거리 기반 + 행동비용)입니다.
- `OdorHold-v4`의 `reward-mode=bio`는 mechanical과 동일한 보상 구조에서 거리 shaping(`exp(-d/sigma_r)`)만 농도 shaping(`bio_reward_scale * c`)으로 바꿉니다.
- `goal/hold` 판정(`d < r_goal`)과 종료 조건은 `mechanical`/`bio` 공통입니다.
- `first.pt`는 `ep100`에서 생성되며, `mid.pt`는 `best.pt`와 `first.pt` 사이 중간 시점에 가장 가까운 스냅샷으로 저장됩니다.
- 중간 학습 중에도 `eval.py` 실행 시 체크포인트가 있으면 `trajectory_first/mid/best.png`를 생성합니다.
- `best_agent.gif`는 evaluation에서 측정한 episode들 중 return이 가장 높은 rollout을 다시 렌더링한 결과입니다.
- Colab/Drive 연결이 끊겨도 학습 산출물은 로컬 복구 경로(`OSL_RECOVERY_DIR` 또는 기본 `/content/osl_recovery`, 비-Colab은 `/tmp/osl_recovery`)에 함께 저장됩니다.

## Output Structure
```text
runs/{agent}_main_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pt
│   ├── final.pt
│   ├── first.pt
│   └── mid.pt
├── plots/
│   ├── returns.png
│   ├── steps_to_goal.png
│   ├── trajectory_first.png    # checkpoint 존재 시
│   ├── trajectory_mid.png      # checkpoint 존재 시
│   ├── trajectory_best.png     # checkpoint 존재 시
│   └── best_agent.gif          # save-gif일 때
├── plot_data/
│   ├── training_metrics.json
│   ├── returns.csv
│   ├── steps_to_goal.csv
│   └── milestones.json
└── config.json
```

## Env v4 Action/Observation
- action: `[v_cmd, omega_cmd, cast_cmd]`
- obs: `[c, mode]`
  - `c`: concentration
  - `mode`: `0=run`, `1=cast`

## Eval Console Metrics
- `Success Rate`
- `Avg Return`
- `Avg Cast Starts`
- `Avg Cast Steps`
- `Cast Step %`
- `Can-Turn %`
