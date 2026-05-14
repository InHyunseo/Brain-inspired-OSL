# CartPole Actor-Critic (PyTorch) + 학습로그/플롯/GIF 저장

Gymnasium `CartPole-v1` 환경에서 **Actor-Critic**(공유 MLP trunk + actor/critic head)로 학습하고, 학습 결과를 `runs/` 아래에 자동 저장하는 단일 스크립트입니다.

- 학습: on-policy, 에피소드 단위(몬테카를로 return)
- 저장: config/metrics.csv/체크포인트(init·mid·final)/학습곡선 PNG/init·mid·final rollout GIF

---

## 1) 요구사항

- Python 3.12+
- 주요 라이브러리
  - `gymnasium`
  - `numpy`
  - `torch`
  - `matplotlib`
  - (GIF 저장용) `imageio` **또는** `Pillow`

설치 예시:
```bash
pip install gymnasium numpy torch matplotlib imageio
````

> `matplotlib.use("Agg")`를 사용하므로 GUI 없이(headless) 서버에서도 이미지 저장이 가능합니다.

---

## 2) 실행 방법

### (A) 기본 실행

```bash
python <스크립트파일명>.py
```

### (B) 파라미터를 바꿔 실행 (추천: import 후 호출)

이 스크립트는 argparse가 없으므로, 아래처럼 파이썬에서 `train()`을 직접 호출하면 편합니다.

```bash
python -c "from <스크립트파일명_확장자없이> import train; train(total_episodes=1000, seed=0, lr=3e-4)"
```

GPU/CPU:

* 기본: CUDA 사용 가능하면 `cuda`, 아니면 `cpu`
* 강제 CPU: `force_cpu=True`

예시:

```bash
python -c "from <스크립트파일명_확장자없이> import train; train(force_cpu=True)"
```

---

## 3) 출력물(결과 저장 구조)

스크립트 위치 기준으로 상대경로 `runs/`에 저장됩니다.

* `SCRIPT_DIR = 현재 스크립트 폴더`
* `DEFAULT_OUT_DIR = SCRIPT_DIR/runs`

실행 1회당 다음 구조가 생성됩니다:

```text
runs/
  cartpole_ac_YYYYmmdd_HHMMSS/        # run_name (기본: 시간 기반 자동 생성)
    config.json                       # 학습 설정 저장
    metrics.csv                       # 에피소드별 로그
    checkpoints/
      init.pt                         # 학습 전 가중치
      mid.pt                          # save_mid_episode 시점
      final.pt                        # 마지막 에피소드
    plots/
      training_curves.png             # steps/return + loss 곡선
    media/
      init.gif                        # init 정책 rollout
      mid.gif
      final.gif
```

---

## 4) 코드 구성 요약

### 4.1 ActorCritic 모델

```python
class ActorCritic(nn.Module):
  body  : (obs_dim -> hidden -> hidden) MLP, Tanh
  actor : hidden -> act_dim (logits)
  critic: hidden -> 1 (V(s))
```

* `forward(x)` → `(logits, value)` 반환
* 행동 샘플링: `Categorical(logits=logits)`에서 샘플

---

### 4.2 Reward Shaping (`shaped_reward`)

CartPole 원래 보상(+1/step) 대신, 상태 기반 shaping을 사용합니다.

* 관측값: `obs = [x, x_dot, theta, theta_dot]`
* 보상(기본 형태):

  * `r = 1 - (theta/theta_th)^2 - 0.01*(x/x_th)^2`
* 종료 페널티/성공 보너스:

  * `terminated`(막대 쓰러짐/범위 이탈)면 `r -= 1`
  * `truncated`(시간 제한 도달)이고 terminated가 아니면 `r += 100`  → “성공” 보상

추가로 학습 안정화를 위해 `reward_scale`(기본 0.01)을 곱해 최종 학습 보상으로 사용합니다.

```python
r = shaped_reward(...) * reward_scale
```

---

### 4.3 학습 루프(`train`)

에피소드 1회 rollout 후, 에피소드 내 모든 step 데이터를 이용해 한 번 업데이트합니다.

에피소드에서 수집하는 것:

* `log_probs[t]` : 선택 행동의 log π(a|s)
* `values[t]`    : critic의 V(s)
* `rewards[t]`   : shaped reward (scaled)
* `entropies[t]` : 정책 엔트로피(탐험 유도)

#### Return(몬테카를로) 계산

```python
G_t = r_t + gamma * G_{t+1}
returns = [G_1, G_2, ..., G_T]
```

#### Advantage 및 정규화

```python
adv = (returns - values).detach()
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

#### Loss 정의

* Actor:

  * `actor_loss = -(log_prob * advantage).mean()`
* Critic:

  * `critic_loss = SmoothL1Loss(values, returns)` (Huber)
* Entropy bonus:

  * `entropy_bonus = entropies.mean()`
* Total:

  ```python
  total_loss = actor_loss + value_coef*critic_loss - entropy_coef*entropy_bonus
  ```

#### 최적화 안정화

* gradient clipping: `clip_grad_norm_(..., max_grad_norm)`

---

## 5) 로그/지표(metrics.csv) 의미

`metrics.csv` 컬럼:

* `episode` : 에피소드 번호(1부터)
* `endured_step` : 해당 에피소드에서 버틴 step 수
* `shaped_return` : shaped reward(이미 `reward_scale` 적용된 값) 누적합
* `success` : `truncated == True` 그리고 `terminated == False`면 1 (시간 제한까지 버팀)
* `actor_loss`, `critic_loss`, `total_loss` : 업데이트 시점 loss 값

터미널 출력(`log_every` 주기):

* 최근 `log_every`개 에피소드 평균 steps/return 및 현재 loss 출력

---

## 6) 플롯/시각화

### 6.1 학습곡선 PNG (`plots/training_curves.png`)

상단:

* `steps/episode` + 이동평균(MA)
* `shaped_return` + 이동평균(MA) (twin axis)

하단:

* `actor_loss`, `critic_loss` + 이동평균(MA)

이동평균 윈도우는 `plot_ma_window`(기본 50)로 조절합니다.

### 6.2 GIF 저장 (`media/*.gif`)

`rollout_video_gif()`가 `render_mode="rgb_array"`로 환경을 열고 프레임을 모아 GIF로 저장합니다.

* 기본은 `deterministic=False`로, 행동을 확률적으로 샘플링한 결과를 저장합니다.
* GIF는 `imageio`가 있으면 우선 사용하고, 없으면 `PIL`로 fallback합니다.
* 동일한 `video_eval_seed`로 init/mid/final을 비교 가능하게 생성합니다.

---

## 7) 주요 하이퍼파라미터(자주 바꾸는 것)

* `total_episodes` : 학습 에피소드 수 (기본 2000)
* `gamma` : discount factor (기본 0.99)
* `lr` : Adam 학습률 (기본 3e-4)
* `value_coef` : critic loss 가중치 (기본 0.5)
* `entropy_coef` : entropy bonus 가중치 (기본 0.01)
* `max_grad_norm` : gradient clipping (기본 0.5)
* `reward_scale` : shaped reward 스케일 (기본 0.01)
* `save_mid_episode` : mid checkpoint 저장 에피소드 (기본 total_episodes//2)
* `log_every` : 출력 주기(기본 50)
* `plot_ma_window` : 이동평균 윈도우(기본 50)
* `video_fps`, `video_eval_seed`

---

## 8) 분석 피봇: trace/probe/patching

이 데모는 본 실험(OSL/connectome 분석)로 가기 전에 분석 프레임워크가 제대로 작동하는지 확인하는 작은 sanity-check 환경으로도 사용합니다. CartPole은 행동 모드가 단순하므로 과학적 발견보다는 다음을 검증하는 목적입니다.

* 학습된 정책의 rollout trace를 episode 단위로 저장할 수 있는가
* MLP hidden activation을 추출해 state variable probe를 할 수 있는가
* hidden activation patching이 실제 행동 성능을 바꾸는가
* CSV/NPZ 결과를 다시 읽어 요약 figure를 만들 수 있는가

### 8.1 실행 순서

GIF 저장까지 원하면 classic-control 렌더 의존성을 설치합니다.

```bash
pip install "gymnasium[classic-control]"
```

학습:

```bash
cd /home/hyunseo/Personal/2d-osl
python -c "from demo.RL_practice.train_cartpole_ac import train; train(total_episodes=2000, seed=0, force_cpu=True, run_name='cartpole_ac_analysis_seed0')"
```

분석:

```bash
python -m demo.RL_practice.analyze_cartpole_ac \
  --run-dir demo/RL_practice/runs/cartpole_ac_analysis_seed0 \
  --checkpoint final \
  --episodes 20 \
  --seed 123 \
  --device cpu
```

그림 생성:

```bash
python -m demo.RL_practice.plot_cartpole_analysis \
  --run-dir demo/RL_practice/runs/cartpole_ac_analysis_seed0
```

### 8.2 분석 출력물

```text
runs/cartpole_ac_analysis_seed0/
  eval/
    trace_final_seed123_ep000.npz
    ...
  analysis/
    baseline_eval.csv
    probe_results.csv
    patch_results.csv
  plots/
    training_curves.png
    analysis_baseline_summary.png
    analysis_probe_r2.png
    analysis_patch_group_delta_steps.png
    analysis_unit_ablation_top.png
    analysis_unit_ablation_scan.png
```

`eval/trace_*.npz`에는 `obs`, `action`, `raw_reward`, `shaped_reward`, `logits`, `value`, `hidden`, `terminated`, `truncated`가 저장됩니다.

`analysis/baseline_eval.csv`는 patch 없는 deterministic eval 결과입니다. CartPole에서는 final policy가 잘 학습되면 `steps`가 500 근처이고 `success`가 1에 가까워야 합니다.

`analysis/probe_results.csv`는 hidden activation으로 `theta`, `theta_dot`을 선형 복원한 R²입니다. CartPole은 이 값들이 observation에 직접 들어 있으므로 R²가 높게 나오는 것이 정상입니다.

`analysis/patch_results.csv`는 hidden activation patching 결과입니다. v1 patch 대상은 `all_hidden`, `first_half`, `second_half`, 그리고 unit-wise zero ablation입니다. `delta_steps`가 음수이면 patch 때문에 baseline보다 빨리 실패했다는 뜻입니다.

### 8.3 현재 관찰된 결과 요약

`cartpole_ac_analysis_seed0` run 기준:

* baseline은 20개 eval seed 모두 500 step 성공(`success_rate=1.0`)으로 완전히 학습됨
* `theta`, `theta_dot` probe R²는 거의 1.0으로 hidden에 상태 정보가 선형적으로 잘 담김
* `all_hidden` patch는 약 `-490 steps`로 정책을 즉시 망가뜨림
* `first_half`/`second_half`를 zero 또는 mean으로 지우는 것은 거의 영향이 없었지만, flip은 치명적이었음
* 단일 hidden unit zero ablation은 대부분 `delta_steps=0`으로, 표현이 단일 unit보다 분산/중복되어 있음을 시사함

해석은 보수적으로 해야 합니다. CartPole은 단순 균형 과제라서 run/cast/turn 같은 행동 모드가 없습니다. 따라서 이 결과는 기전 발견보다는 분석 plumbing 검증입니다.

### 8.4 본 실험으로 옮길 때 바꿀 부분

이 프레임워크는 본 실험에 그대로 가져가되, 아래 부분만 환경/정책에 맞게 교체하면 됩니다.

* `trace dump`: CartPole의 `obs/action/reward/hidden` 대신 OSL의 trajectory, sensor readings, action channels, event labels, connectome state를 저장
* `probe`: `theta/theta_dot` 대신 bearing, distance, sensor asymmetry, `dlog concentration`, event label 등을 target으로 사용
* `patch`: MLP hidden index 대신 connectome cell type/group index를 사용하고, zero/mean/flip patch를 actor forward에 적용
* `figures`: baseline step/success 대신 OSL success, path efficiency, event ratio, cast frequency, patch-induced behavior shift를 그림으로 요약
* `dynamics`: CartPole v1에서는 제외했지만, recurrent/connectome state가 있는 본 실험에서는 segment-conditioned Jacobian/eigenvalue 분석을 추가

---

## 9) 참고/주의사항

* Gymnasium 버전에 따라 `render_mode="rgb_array"` 동작이 다를 수 있습니다. GIF가 비어 있으면 gymnasium 버전과 렌더 옵션을 확인하세요.
* shaped reward는 원래 CartPole 보상과 스케일이 다르므로, `reward_scale`을 너무 크게 하면 학습이 불안정해질 수 있습니다.
* `success`는 “시간 제한까지 버틴 경우(truncated)”를 의미합니다(즉, Pole이 쓰러져 종료(terminated)된 것은 실패로 봄).

---


