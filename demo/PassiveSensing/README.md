# OSL (Odor Source Holding) — DQN 데모

중앙 `(0,0)`에 냄새 소스가 있고, 멀어질수록 농도가 감소하는 2D 환경에서  
**좌/우 농도 센서 2개만**으로 **소스 근처에 오래 머무르게** 학습하는 DQN 데모입니다.  
로봇은 **선속도 고정**, **각속도만 이산 선택**합니다.

---

## 파일 구성

- `odor_env.py`  
  Gymnasium 환경. 내부 상태는 `(x, y, theta)`이고 관측은 `(cL, cR)`를 **최근 N스텝 스택**해서 사용합니다.  
  보상은 “소스에 가까울수록 +, goal 반경 안이면 보너스, 회전 패널티” 형태.

- `train.py`  
  DQN 학습 스크립트. `runs/<run_name>/` 아래에 결과 저장:
  - `checkpoints/ init.pt, mid.pt, final.pt`
  - `media/init.gif, mid.gif, final.gif`
  - `metrics.csv`, `config.json` (env_kwargs 포함)

- `eval.py`  
  run 폴더의 **init/mid/final 모델을 불러와** greedy(ε=0)로 rollout 후  
  **trajectory(궤적) plot**을 `media/traj_*.png`로 저장 (옵션: `traj_*.json`).

---

## 모델 구조(요약)

입력: 스택된 센서값 (예: `stack_n=4`면 `[cL,cR]x4 → 8차원`)  
출력: 각 행동(7개 각속도)에 대한 Q값

```

(obs_dim) → FC(256) → ReLU → FC(256) → ReLU → FC(act_dim=7)

````

---

## 실행

### 1) 학습
```bash
python train.py
````


예시(에피소드/노이즈):

```bash
python train.py --total-episodes 600 --sensor-noise 0.01
python train.py --run-name osl_demo --total-episodes 600 --sensor-noise 0.01
```

### 2) 평가 + 궤적 저장 (init/mid/final 모두)

```bash
python eval.py --run_dir runs/odor_dqn_20260130_184544 --episodes 50
```

궤적 raw 데이터도 저장:

```bash
python eval.py --run_dir runs/odor_dqn_20260130_184544 --episodes 50 --save_json
```

저장 결과:

* `runs/<run_name>/media/traj_init.png`
* `runs/<run_name>/media/traj_mid.png`
* `runs/<run_name>/media/traj_final.png`
  (+ 옵션) `traj_*.json`

---

## 메모

* 평가에서 greedy(ε=0)는 “랜덤 탐험 없이 항상 `argmax Q` 행동만 선택”을 뜻합니다.
* `sensor_noise=0.01` 등 env 파라미터는 `config.json`의 `env_kwargs`로 저장되어 평가에도 동일 적용됩니다.
