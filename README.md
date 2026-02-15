# OdorHold DRQN Project

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
python3 main.py --total-episodes 10000
```

예시 (환경 파라미터 변경):
```bash
python3 main.py \
  --total-episodes 2000 \
  --wind-x 0.5 \
  --src-x -1.0 \
  --src-y 0.0
```

### 2. Train Only
평가 없이 학습만 수행합니다.

```bash
python3 train.py --total-episodes 600 --out-dir runs --run-name my_run
```

### 3. Eval Only
기존 run 결과를 불러와 성능 측정과 시각화를 다시 수행합니다.

```bash
python3 eval.py --run-dir runs/drqn_main_YYYYMMDD_HHMMSS --episodes 20 --save-gif
```

## Common Arguments
`main.py` 기준 자주 쓰는 인자:
- `--total-episodes` (default: `600`)
- `--lr` (default: `1e-4`)
- `--rnn-hidden` (default: `147`)
- `--batch-size` (default: `64`)
- `--seq-len` (default: `16`)
- `--src-x`, `--src-y` (default: `0.0`, `0.0`)
- `--wind-x` (default: `0.0`)
- `--eval-episodes` (default: `10`)
- `--force-cpu`

`eval.py` 추가 인자:
- `--ckpt` (default: 자동 선택: `checkpoints/best.pt`가 있으면 사용, 없으면 `checkpoints/final.pt`)
- `--seed-base` (default: `20000`)
- `--save-gif` / `--no-save-gif` (default: `--save-gif`)

## Output Structure
`main.py` 실행 시 결과는 `runs/drqn_main_YYYYMMDD_HHMMSS/`에 저장됩니다.

```text
runs/drqn_main_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pt
│   └── final.pt
├── plots/
│   ├── returns.png
│   └── steps_to_goal.png
├── config.json
├── eval_result.png
└── best_agent.gif
```

참고:
- `best.pt`는 학습 중 최고 episode return 갱신 시 저장됩니다.
- `steps_to_goal.png`는 에피소드별 source 최초 도달 step(작을수록 빠른 도달)을 보여줍니다.
- `eval.py`는 `best.pt`가 있으면 우선 사용하고, 없으면 `final.pt`를 사용합니다.
- `best_agent.gif`는 선택된 체크포인트(`best.pt` 또는 `final.pt`)로 `seed-base` 한 에피소드를 롤아웃해 저장한 영상입니다.
