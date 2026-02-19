# Odor Source Localization Project

`main.py` 하나로 DRQN 학습, 평가, 시각화(PNG/GIF)를 순차 실행하는 프로젝트입니다.

## Environment
- OS: Ubuntu 22.04
- Python: 3.10.12

## Setup
```bash
cd ~/osl_project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. End-to-End (Train + Eval + Plot/GIF)
학습 후 같은 run 디렉터리에서 평가와 시각화를 자동으로 수행합니다.

```bash
python3 main.py --total-episodes 6000 --eps-decay-steps 1000
```

예시 (환경 파라미터 변경):
```bash
python3 main.py \
  --total-episodes 20000 \
  --wind-x 1.0 \
  --src-x -2.0 \
  --sigma-c 1.2 \
```

예시 (DQN으로 학습):
```bash
python3 main.py --agent-type dqn --total-episodes 20000
```

예시 (연속 제어 RSAC + env v4):
```bash
python3 main.py --agent-type rsac --env-id OdorHold-v4 --total-episodes 20000
```
`OdorHold-v4`는 하이브리드 액션 `[v_cmd, omega_cmd, cast_cmd]`를 사용합니다.
`cast_cmd`는 정책이 확률적으로 샘플링해 결정하며, cast 중에는 v3처럼 L/R/L/R 고정 active sensing을 수행합니다.

### 2. Train Only
평가 없이 학습만 수행합니다.

```bash
python3 train.py --total-episodes 6000 --out-dir runs
```

### 3. Eval Only
기존 run 결과를 불러와 성능 측정과 시각화를 다시 수행합니다.

```bash
python3 eval.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --episodes 20 --save-gif
```

### 4. Replot Only
학습 그래프는 저장된 `plot_data`로 재생성하고, trajectory는 체크포인트로 다시 롤아웃해 재생성합니다.

```bash
python3 replot.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --target all
```

부분 재생성:
```bash
python3 replot.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --target train  # returns/steps
python3 replot.py --run-dir runs/{agent}_main_YYYYMMDD_HHMMSS --target eval   # trajectory_first/mid/best
```

## Common Arguments
`main.py` 기준 자주 쓰는 인자:
- `--total-episodes` (default: `600`)
- `--agent-type` (default: `drqn`, choices: `drqn`, `dqn`, `rsac`)
- `--lr` (default: `1e-4`)
- `--lr-actor` (default: `3e-4`, RSAC)
- `--lr-critic` (default: `3e-4`, RSAC)
- `--lr-alpha` (default: `3e-4`, RSAC)
- `--gamma` (default: `0.99`, RSAC)
- `--tau` (default: `0.005`, RSAC)
- `--rnn-cell` (default: `gru`, choices: `gru`, `rnn`)
- `--rnn-hidden` (default: `147`)
- `--dqn-hidden` (default: `256`)
- `--batch-size` (default: `64`)
- `--seq-len` (default: `16`)
- `--src-x`, `--src-y` (default: `0.0`, `0.0`)
- `--wind-x` (default: `0.0`)
- `--sigma-c` (default: `1.0`)
- `--eval-episodes` (default: `10`)
- `--save-milestones` / `--no-save-milestones` (default: `--save-milestones`)
- `--first-milestone-ep` (default: `100`)
- `--milestone-every` (default: `10`)
- `--force-cpu`

`eval.py` 추가 인자:
- `--ckpt` (default: 자동 선택: `checkpoints/best.pt`가 있으면 사용, 없으면 `checkpoints/final.pt`)
- `--seed-base` (default: `20000`)
- `--save-gif` / `--no-save-gif` (default: `--save-gif`)
- `--plot-milestones` / `--no-plot-milestones` (default: `--plot-milestones`)

## Output Structure
`main.py` 실행 시 결과는 `runs/{agent}_main_YYYYMMDD_HHMMSS/`에 저장됩니다.
(`{agent}`는 `drqn`, `dqn`, `rsac`)

```text
runs/{agent}_main_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pt
│   ├── first.pt
│   ├── mid.pt
│   └── final.pt
├── plots/
│   ├── returns.png
│   ├── steps_to_goal.png
│   ├── trajectory_first.png
│   ├── trajectory_mid.png
│   ├── trajectory_best.png
│   └── best_agent.gif
├── plot_data/
│   ├── training_metrics.json
│   ├── returns.csv
│   ├── steps_to_goal.csv
│   └── milestones.json
├── config.json
```

참고:
- `best.pt`는 학습 중 최고 episode return 갱신 시 저장됩니다.
- `steps_to_goal.png`는 에피소드별 source 최초 도달 step(작을수록 빠른 도달)을 보여줍니다.
- `eval.py`는 `best.pt`가 있으면 우선 사용하고, 없으면 `final.pt`를 사용합니다.
- `best_agent.gif`는 선택된 체크포인트(`best.pt` 또는 `final.pt`)로 `seed-base` 한 에피소드를 롤아웃해 저장한 영상입니다.
