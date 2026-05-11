# Odor Source Localization (Brain-Inspired RL)

2D OSL (단일 농도 센서, cast/run hybrid action). PPO + recurrent connectome 정책이
주 실험 경로, RSAC / DRQN은 비교군.

## Project Goals
- 단일 센서 + 부분관측 환경에서 source 탐색/유지
- cast 행동의 학습 가능성 검증
- ORN/PN/LN/KC/MBON 5-population connectome 정책

## Code Status

### Environments (`src/envs/osl_env_2d.py`)
- `StaticEnv` — Gaussian plume (no turbulence)
- `DynamicEnv(noise_coef)` — turbulent plume, 100×100 cached field, gaussian-blur bumps
  - 두 환경 모두 obs `[c, did_cast]`, action `Box([v, omega, cast])`
  - 4-step cast lock (scan pattern `+π/2, -π/2, +π/2, -π/2`)

### Agents (`src/agents/`)
- **`PPOAgent`** — sb3 `RecurrentPPO` + 4-phase 커리큘럼 (static → coef 0.3 → 0.6 → 1.0).
  Connectome feature extractor + LSTM policy. 16-way `SubprocVecEnv` + `VecNormalize`.
- **`RSACAgent`** — recurrent SAC, hybrid actor (Gaussian `[v, omega]` + Bernoulli `cast`).
  Backbone 3종: `gru` / `connectome` / `mlp`.
- **`DRQNAgent`** — discrete `{RUN, CAST, TURN_L, TURN_R}` action set + 내부 action adapter.
  `recurrent` 플래그로 GRU(DRQN) ↔ MLP(DQN) 토글.

### Networks (`src/models/networks.py`)
- `GRUActor`, `ConnectomeActor`, `MLPActor` — RSAC actor 백본 3종.
- `ConnectomeExtractor` — sb3 `BaseFeaturesExtractor` 구현, PPO 전용.
- `QCritic` — RSAC twin-critic 컴포넌트 (GRU).
- `QNet` — DRQN/DQN 통합 Q-net (`recurrent` 플래그).

### Connectome (PPO `ConnectomeExtractor` / RSAC `ConnectomeActor` 공통)
- ORN:PN:LN:KC:MBON = 24:7:4:54:1 (총합 90의 배수, default `features_dim=180`)
- 1 외부 step당 4 내부 tanh 업데이트
- `ORN ← W_oto(ORN) + W_lto(LN) + x_t`
- `PN  ← W_otp(ORN') + W_ltp(LN) + W_ptp(PN)`
- `LN  ← W_otl(ORN') + W_ptl(PN') + W_ltl(LN)`
- `KC  ← W_ktk(KC) + W_mtk(MBON) + W_ptk(PN')`
- `MBON ← W_ktm(KC')`

## Entry Points
- `train.py` — `--agent-type {ppo, rsac, drqn}` 분기
- `eval.py` — train과 동일 분기, `--run-dir` 필요
- `main.py` — train + eval 한 번에
- `replot.py` — `plot_data/*.json`에서 PNG 재생성 (RSAC/DRQN)
- `ipynb/PPO_framework.ipynb` — 노트북 기준선 (이 레포의 진리표)
- `ipynb/DRQN_framework.ipynb` — DRQN/DQN을 같은 스타일로 단독 실행

## Removed (legacy)
- `OdorHold-v3` / `OdorHold-v4` env, connectome v1 (3:1:4:1), 별도의
  `DQNAgent`/`dqn_agent.py` — 모두 제거. DQN 기능은 `drqn_agent.py`로 흡수
  (`--no-drqn-recurrent`).

## Reference
- 실행 명령/옵션/출력 구조: `MANUAL.md`
