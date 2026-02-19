# Odor Source Localization Research (Brain-Inspired RL)

이 저장소는 단일 농도 센서 기반의 active sensing 후각 탐색을,
뇌영감(connectome/도파민 예측오차 해석) 관점에서 강화학습으로 검증하는 연구 코드입니다.

## Research Goal
- 단일 센서 + 부분관측 환경에서 source localization/holding 달성
- 행동 레벨에서 active sensing(casting)이 학습되는지 검증
- TD error를 도파민성 prediction-error의 계산적 대응(computational analogue)로 해석
- 최종적으로 connectome 제약(masked recurrent core)과 성능/행동 패턴의 관계 분석

## Current Status
- Baseline: `DRQN/DQN + OdorHold-v3` 유지됨
- New path: `RSAC + OdorHold-v4` 추가됨
- `OdorHold-v4`는 하이브리드 action 사용
  - `action = [v_cmd, omega_cmd, cast_cmd]`
  - `cast_cmd`는 정책이 Bernoulli로 샘플링
  - cast 중에는 v3 스타일 고정 샘플링(`L/R/L/R`, 정지)
- 관측은 v3 스타일 1-step 유지
  - `obs = [c, mode]` (`mode: 0=run, 1=cast`)

## Recommended Experimental Order
1. v3 baseline 성능 고정 (success, return, step-to-goal)
2. v4 + RSAC 학습 안정화
3. `GRU vs vanilla RNN` ablation (`--rnn-cell gru|rnn`)
4. connectome mask 삽입 실험

## Key Hypotheses
- 불확실 구간에서 cast 선택 빈도가 증가한다.
- cast를 막으면 성공률/도달시간이 악화된다.
- recurrent cell 종류와 connectome 제약이 active sensing 패턴에 영향을 준다.

## Metrics to Track
- Task: success rate, avg return, reach step
- Behavior: cast rate, cast burst length, 경로 길이/진동, |omega|
- Learning: TD error 통계, alpha(entropy temperature), Q 분포

## Document Map
- 실행/인자/출력 구조 매뉴얼: `MANUAL.md`
- 코드 엔트리포인트: `main.py`, `train.py`, `eval.py`
- 환경: `src/envs/odor_env.py`, `src/envs/odor_env_v4.py`
- 모델: `src/models/networks.py`
- 에이전트: `src/agents/`

## Quick Start
```bash
cd ~/osl_project
python3 main.py --agent-type rsac --env-id OdorHold-v4 --total-episodes 20000
```

