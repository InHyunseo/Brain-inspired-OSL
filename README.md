# Odor Source Localization (Brain-Inspired RL)

단일 농도 센서 기반 odor source localization/holding 문제를,
brain-inspired 관점(부분관측, active sensing, prediction error 해석)으로 실험하는 코드베이스입니다.

## Research Focus
- 단일 센서 + 부분관측 환경에서 안정적인 source 탐색/유지
- active sensing(casting) 행동의 학습 여부 검증
- actor-critic의 critic TD error를 dopamine-like prediction error의 computational analogue로 해석
- 이후 connectome 제약(masked recurrent core)으로 확장

## Current Implementations
- `OdorHold-v3` + `DRQN/DQN`
  - 이산 action 기반 기존 baseline
- `OdorHold-v4` + `RSAC` (Recurrent SAC, hybrid action)
  - action: `[v_cmd, omega_cmd, cast_cmd]`
  - cast 시작은 정책이 결정, cast 중에는 v3 스타일 고정 샘플링(`L/R/L/R`, 정지)
  - 기본 설정은 hard constraint(`cast` 이후에만 turn 허용)

## Current Workflow
1. hard-constraint + GRU로 수렴 baseline 확보
2. GRU 안정화 후 soft constraint 실험
3. `GRU vs vanilla RNN` ablation
4. connectome mask 삽입

## Key Metrics
- Task: success rate, avg return, step-to-goal
- Behavior: cast start count, cast step ratio, turn-available ratio, trajectory shape
- Learning: TD error, alpha(entropy temperature), critic/actor loss trend

## Quick Start
```bash
cd ~/osl_project
python3 main.py --agent-type rsac --env-id OdorHold-v4 --total-episodes 20000
```

## Documents
- 실행/옵션/출력 구조: `MANUAL.md`
- 진입점: `main.py`, `train.py`, `eval.py`, `replot.py`
