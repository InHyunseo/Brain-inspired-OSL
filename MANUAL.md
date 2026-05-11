# Manual

## Environment
- Ubuntu 22.04 (WSL2 가능), Python 3.10
- `sb3-contrib`의 `RecurrentPPO`는 gymnasium 0.29 + numpy<2 에서 안정

## Setup
```bash
cd ~/Personal_Research/OSL
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Commands

### PPO (기본, 4-phase curriculum)
```bash
python3 train.py --agent-type ppo
```
phase 진행 후 `runs/ppo_main_*/checkpoints/phase{i}_{kind}.zip` + `*_vnorm.pkl` 저장.

#### 더 짧은 1-phase 학습 (디버깅)
```bash
python3 train.py --agent-type ppo \
  --phases '[["static", 50000]]' --n-envs 4
```

### RSAC
```bash
python3 train.py --agent-type rsac \
  --rsac-actor-backbone connectome \
  --connectome-hidden 180 --connectome-steps 4 \
  --total-episodes 20000
```
- `--rsac-actor-backbone {gru, connectome, mlp}`
- `--connectome-hidden`은 90의 배수 (default 180)

### DRQN / DQN
```bash
# DRQN (recurrent)
python3 train.py --agent-type drqn --total-episodes 20000

# vanilla DQN (no recurrence)
python3 train.py --agent-type drqn --no-drqn-recurrent --total-episodes 20000
```

### End-to-end (train + eval)
```bash
python3 main.py --agent-type ppo
```

### Eval only
```bash
python3 eval.py --run-dir runs/ppo_main_YYYYMMDD_HHMMSS --episodes 50
```
PPO는 `final.zip` + `final_vnorm.pkl`을 로드, elite seed 탐색 후 GIF 렌더.
RSAC/DRQN은 `best.pt` (없으면 `final.pt`) 로드, 베스트 return episode의 GIF 렌더.

### Replot
```bash
python3 replot.py --run-dir runs/<agent>_main_YYYYMMDD_HHMMSS
```

## Common Arguments (`train.py`)

### 공통
- `--agent-type {ppo, rsac, drqn}` (default `ppo`)
- `--seed`, `--force-cpu`, `--out-dir`, `--run-name`

### PPO
- `--phases` (JSON list, default 4-phase curriculum)
- `--n-envs` (default 16), `--features-dim` (default 180, 90의 배수)
- `--ppo-lr` (3e-4), `--ppo-batch-size` (256), `--ppo-n-steps` (128), `--ppo-ent-coef` (0.01)
- `--tb-log` (default `<run_dir>/tb`)

### RSAC
- `--rsac-actor-backbone {gru, connectome, mlp}` (default `gru`)
- `--connectome-hidden` (default 180, 90의 배수), `--connectome-steps` (default 4)
- `--rnn-hidden` (default 147)
- `--lr-actor / --lr-critic / --lr-alpha` (default 3e-4)
- `--gamma 0.99`, `--tau 0.005`
- `--batch-size 128`, `--seq-len 16`, `--buffer-size 150000`, `--learning-starts 5000`
- `--rsac-env-kind {static, dynamic_0.3, dynamic_0.6, dynamic_1.0}` (default `dynamic_1.0`)

### DRQN
- `--drqn-recurrent` / `--no-drqn-recurrent` (default on, recurrent=True → DRQN)
- `--rnn-hidden 147`, `--lr 1e-4`, `--gamma 0.99`
- `--eps-start 1.0`, `--eps-end 0.05`, `--eps-decay-steps 4000`
- `--target-update-every 20`
- `--drqn-env-kind {static, dynamic_0.3, dynamic_0.6, dynamic_1.0}`

### Eval-side
- `--eval-episodes` / `--episodes` (alias), `--seed-base`, `--ckpt`
- `--save-gif` / `--no-save-gif`
- `--noise-coef` (PPO eval env의 DynamicEnv 강도, default 1.0)

## Env API
- obs: `[c, mode]`
  - `c`: concentration
  - `mode`: 직전 step에서 `did_cast` 여부 (1.0=cast, 0.0=run)
- action: `Box([v, omega, cast_prob], shape=(3,))`
  - `cast_prob`는 `np.rint(np.clip(., 0, 1))` 후 1이면 4-step lock cast 시작
  - cast 시작 후 4 step 동안 `+π/2, -π/2, +π/2, -π/2` 스캔
- reward: `0.5 * c - [cast_penalty or motion_penalty] + b_hold(in_goal) - b_oob(oob)`
- 종료: `goal_hold_count >= 20` 성공, `|x| > L or |y| > L` 실패, `step >= 300` truncated

## Output Structure (PPO)
```
runs/ppo_main_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── phase0_static.zip
│   ├── phase0_static_vnorm.pkl
│   ├── phase1_dynamic_0.3.zip
│   ├── phase1_dynamic_0.3_vnorm.pkl
│   ├── ...
│   ├── final.zip
│   └── final_vnorm.pkl
├── plots/best_agent.gif
├── tb/                       # TensorBoard event files
└── config.json
```

## Output Structure (RSAC / DRQN)
```
runs/{agent}_main_YYYYMMDD_HHMMSS/
├── checkpoints/{best,first,final}.pt
├── plots/{returns.png, steps_to_goal.png, best_agent.gif}
├── plot_data/{training_metrics.json, returns.csv, steps_to_goal.csv}
└── config.json
```

## Notes
- `sb3-contrib`의 `RecurrentPPO`는 numpy<2 권장 — `requirements.txt`는 `numpy==1.26.4`로
  설정. 환경에 따라 sb3 2.3+가 numpy 2를 지원하면 상향 가능.
- `connectome_hidden=180`은 `k=2`이므로 MBON=2. 더 큰 표현력이 필요하면 `360`(k=4) 권장.
- DRQN/DQN은 노트북에서도 동일 결과를 얻고 싶다면 `ipynb/DRQN_framework.ipynb` 사용.
