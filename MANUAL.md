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
- `--env-id`: default `OdorHold-v3`
- `--agent-type`: `drqn | dqn | rsac` (default `drqn`)
- `--total-episodes`: default `600`
- `--seed`: default `42`
- `--force-cpu`
- `--spawn-mode`: `legacy | balanced` (default `balanced`, v3 전용)
- `--reward-mode`: `mechanical | bio` (default `mechanical`)

### RSAC-related
- `--lr-actor`: default `3e-4`
- `--lr-critic`: default `3e-4`
- `--lr-alpha`: default `3e-4`
- `--gamma`: default `0.99`
- `--tau`: default `0.005`
- `--rnn-cell`: `gru | rnn` (default `gru`)
- `--rnn-hidden`: default `147`
- `--batch-size`: default `128`
- `--seq-len`: default `16`
- `--buffer-size`: default `150000`
- `--learning-starts`: default `5000`

### Milestone / Eval
- `--save-milestones` / `--no-save-milestones` (default save)
- `--milestone-every`: default `10`
- `--eval-episodes`: default `10`
- `--seed-base`: default `20000`
- `--save-gif` / `--no-save-gif` (default save)
- `--plot-milestones` / `--no-plot-milestones` (default plot)

## Notes
- `RSAC`에서는 `--eps-start/--eps-end/--eps-decay-steps`가 실질적으로 사용되지 않습니다.
- `OdorHold-v3`의 `spawn-mode=balanced`는 소스가 경계로 이동할 때 스폰 반경 분포 왜곡을 줄이기 위한 보정 샘플러입니다.
- `OdorHold-v4`의 `reward-mode=mechanical`은 전통 RL shaping(거리 기반 + 행동비용)입니다.
- `OdorHold-v4`의 `reward-mode=bio`는 mechanical과 동일한 보상 구조에서 거리 shaping(`exp(-d/sigma_r)`)만 농도 shaping(`c`)으로 바꿉니다.
- `goal/hold` 판정(`d < r_goal`)과 종료 조건은 `mechanical`/`bio` 공통입니다.
- `first.pt`는 `ep100`에서 생성되며, 이후 중간 체크포인트는 `ep100`부터 `milestone-every` 주기에 맞춰 저장됩니다.
- 중간 학습 중에도 `eval.py` 실행 시 체크포인트가 있으면 `trajectory_first/mid/best.png`를 생성합니다.

## Output Structure
```text
runs/{agent}_main_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pt
│   ├── final.pt
│   ├── first.pt                # save-milestones일 때
│   └── mid.pt                  # save-milestones일 때
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
│   └── milestones.json         # save-milestones일 때
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
