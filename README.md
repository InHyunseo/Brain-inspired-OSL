# Odor Source Localization (Brain-Inspired RL)

단일 농도 센서 기반 odor source localization/holding을 실험하는 RL 코드베이스입니다.  
핵심 관심사는 부분관측 환경에서의 active sensing(casting), recurrent memory, TD error 해석입니다.

## Project Goals
- 단일 센서 + 부분관측 환경에서 source 탐색/유지
- cast 행동의 학습 가능성 검증
- critic TD error를 dopamine-like prediction error의 computational analogue로 해석
- 최종적으로 connectome 제약 구조로 확장

## Current Code Status

### Environments
- `OdorHold-v3`
  - 이산 action 기반 baseline
  - spawn sampler: balanced 고정
- `OdorHold-v4`
  - action: `[v_cmd, omega_cmd, cast_cmd]`
  - observation: `[c, mode]` (`mode`: run/cast)
  - cast 시작은 정책이 결정
  - cast 수행은 현재 고정 패턴(`L/R/L/R`, 정지)으로 구현
  - 기본 turn 제약: cast 이후 turn 허용(hard constraint)
  - spawn sampler: balanced 고정

### Agents
- `DRQN / DQN` (v3 baseline)
- `RSAC` (v4 기본)
  - recurrent hybrid actor (`[v, omega]` Gaussian + `cast` Bernoulli)
  - twin critic(Q1/Q2) + target critic
  - recurrent critic 고정

### Reward Modes in v4
| Mode | Base Shaping | Goal/Hold Criterion | Purpose |
|---|---|---|---|
| `mechanical` | `exp(-d/sigma_r)` | `d < r_goal` | 전통 RL shaping baseline |
| `bio` | `c` | `d < r_goal` | chemotaxis-inspired shaping |

정리하면 현재 `mechanical`과 `bio`의 핵심 차이는 base shaping 항(`distance` vs `concentration`)입니다.

## v3 vs v4 (Core Differences)
| Item | `OdorHold-v3` | `OdorHold-v4` |
|---|---|---|
| Action space | Discrete 4 actions (`RUN/CAST/TURN_L/TURN_R`) | Hybrid 3 actions (`[v_cmd, omega_cmd, cast_cmd]`) |
| Motion control | 고정 전진속도 + 이산 회전(`cast_turn`) | 연속 속도/각속도 명령 + 가속도 제한 |
| Cast behavior | 고정 4-step cast(`L/R/L/R`) 후 turn 선택 필요 | cast 시작은 정책 결정, cast 수행은 고정 4-step(`L/R/L/R`) |
| Turn constraint | cast 이후 `need_turn` 상태에서만 turn 진행 | 기본 hard constraint: cast 이후 turn window에서만 turn 허용 |
| Observation | `[c, mode] * stack_n` (env stacking) | `[c, mode]` (no env stacking) |
| Reward | 거리 shaping 단일 모드(`exp(-d/sigma_r)`) | `mechanical`(거리) / `bio`(농도) |
| Spawn sampler | balanced 고정 | balanced 고정 |
| Typical agent | `DRQN/DQN` baseline | `RSAC` |

## Metrics
- Task: success rate, avg return, step-to-goal
- Behavior: cast start count, cast step ratio, turn-available ratio, trajectory
- Learning: TD error(`td_abs`), alpha(entropy temperature), actor/critic loss

## Code Entry Points
- Pipeline: `main.py`
- Train: `train.py`
- Eval: `eval.py`
- Replot: `replot.py`
- Env(v4): `src/envs/odor_env_v4.py`
- Networks: `src/models/networks.py`

## Current Limitations
- cast 루틴 일부(`L/R/L/R`)가 하드코딩되어 있어 완전 end-to-end 정책은 아님
- 학습 규칙은 SAC(backprop/Adam) 기반이라 생물학적 local plasticity와는 차이가 있음

## Reference Doc
- 실행 명령/옵션/출력 구조 상세: `MANUAL.md`

## Next Research Flow (Agreed)
1. 변경된 `bio` 모드 baseline을 먼저 안정화하고 파라미터 튜닝
2. cast 하드코딩을 단계적으로 완화/대체 (hard -> soft -> learned timing)
3. 네트워크 구조를 bio-inspired 방향으로 변경 (예: vanilla RNN/구조적 priors, DAN-like TD modulation 해석 강화)
4. 최종적으로 connectome 제약 네트워크로 대체

실험 원칙:
- 한 번에 한 축만 변경
- 각 단계는 multi-seed로 비교
- 이전 안정 체크포인트를 기준선으로 유지 후 다음 단계 진행
