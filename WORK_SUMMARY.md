# OSL 작업 정리

## 핵심 작업물

### 학습 및 실행 파이프라인

- `main.py`, `train.py`, `eval.py` 기반의 OSL 학습, 평가, 실행 파이프라인 구축
- 학습 설정을 `src/utils/config.py`, `src/utils/factory.py`로 분리해 agent, environment, model 구성을 재사용 가능하게 정리
- checkpoint 저장, best model 저장, 학습 중단 후 재개 기능, seed 고정 및 재현성 관리 기능 추가
- Colab 실행용 notebook과 로컬 실행 cell을 구성해 실험 재현성을 높임

### OSL 환경 및 보상 설계

- 초기 `odor_env.py` 기반 환경에서 `src/envs/osl_env.py` 중심의 2D OSL 환경으로 개편
- odor field, geometry, event, parallel runner를 모듈화
- source spawn, 시작 위치 범위, 센서 위치, head casting, source clipping, radius, curriculum을 반복적으로 조정
- 거리 기반 reward에서 농도 기반 bio reward 및 dense reward를 도입
- Gaussian local noise와 spawn 위치 반영을 추가해 odor field 설정을 개선
- 시간차 정보를 observation에 포함해 active sensing 입력을 확장

### Agent 및 모델

- DRQN 기반 초기 agent 구현
- MLP DQN agent 추가
- RSAC 및 SAC 계열 agent 구현과 튜닝
- PPO agent 구현 및 TensorBoard logging, entropy logging, cast logging 추가
- GRU 기반 policy/backbone 구성
- connectome 기반 actor/policy 구조 추가 및 반복 수정
- recurrent critic과 stateless critic 구조를 실험하고 선택 가능하게 정리
- baseline chemotaxis 모델 구현 및 tuning

### 분석 프레임워크

- `analysis/osl2d` 분석 파이프라인 구축
- evaluation dump, trajectory labeling, segmentation, latent visualization, probe, neuron analysis, Jacobian analysis, fixed point analysis, ablation, batch runner 구성
- PPO-GRU active sensing top-k PCA 분석 스크립트 추가
- PPO-GRU noise sweep 평가 스크립트 추가
- CartPole 기반 RL 분석 예제와 plotting 도구를 demo에 통합
- 분석 방법론과 수식 정리를 `analysis/METHODS_MATH.md`, `analysis/methodology.md`에 정리

### 시각화 및 결과 생성

- return, steps-to-goal, trajectory plot 저장 구조 구성
- 학습 중단 시점에도 plot이 생성되도록 개선
- replot pipeline을 추가해 저장된 csv/json 결과에서 시각화를 다시 생성할 수 있게 함
- GIF 생성 및 평가 episode 중 reward가 가장 높은 seed를 선택하는 방식 추가
- curriculum field 시각화 스크립트 추가
- baseline noise sweep, trajectory, presentation/poster용 그림 생성 notebook 정리

### Notebook 및 Demo

- `notebooks/baseline.ipynb`: baseline chemotaxis 실험
- `notebooks/ppo_connectome.ipynb`: PPO-connectome 실험
- `notebooks/ppo_gru.ipynb`: PPO-GRU 실험
- `demo/DRQN`: DRQN minimal demo
- `demo/PassiveSensing`: passive sensing 실험 demo
- `demo/RL_practice`: CartPole actor-critic 및 분석 demo

### Connectome 자산

- `assets/connectome/metadata.csv`
- `assets/connectome/weights.csv`
- connectome policy 및 valence/connectome 관련 실험 구조 구성

## 날짜별 작업 내역

### 2026-02-15

- Modular DRQN 기반 OSL 초기 구조 구축
- `eval.py`, `main.py`, `train.py`, `src/agents/drqn_agent.py`, `src/envs/odor_env.py`, `src/models/networks.py`, `src/utils/buffer.py`, `src/utils/plotter.py` 구성
- default parameter 조정
- `best.pt` 저장 기능 추가
- 프로젝트 명칭을 Odor Source Localization으로 정리
- MLP 기반 DQN agent 추가
- plot 저장 경로를 `plots` 중심으로 정리
- 긴 학습 episode에 맞춰 epsilon decay step 조정

### 2026-02-16

- 학습이 중간에 멈춰도 plot이 생성되도록 `train.py` 개선
- Colab 실행용 notebook 구성 및 실행 URL 정리

### 2026-02-17

- GIF RGB render와 trajectory plot의 표현을 맞춤
- return plot 오류 수정
- plot style 변경
- 학습 후 결과를 csv/json으로 저장하고 `replot.py`로 다시 그릴 수 있는 구조 추가
- odor field parameter 중 `sigma_r` 조정

### 2026-02-18

- 당시 성능이 가장 좋았던 hyperparameter 조합 반영
- README 및 실행 설정 정리

### 2026-02-19

- DRQN 결과 산출물 저장
- 연속 action 출력을 위한 RSAC 계열 구조 도입
- `src/agents/rsac_agent.py`, `src/envs/odor_env_v4.py` 추가
- RSAC 실험 결과, checkpoint, trajectory, plot 산출

### 2026-02-20

- angular acceleration 범위를 조정해 cast 동작을 완화
- entropy가 초반에 급격히 감소하지 않도록 조정
- critic loss를 MSE에서 Huber loss로 변경
- critic network를 MLP 구조로 변경
- Q값 전달 버그 수정
- seed 고정 및 재현성 관리 유틸 추가

### 2026-02-22

- DQN, DRQN, RSAC agent와 buffer 관련 bug fix
- cast penalty 설정 조정

### 2026-02-23

- milestone 저장 구조 단순화
- source 위치 이동에 따른 clipping 심화 문제를 spawn mode balanced 방식으로 완화
- hyperparameter default 조정
- 거리 기반 reward를 농도 기반 bio reward로 전환
- goal 도달 판정은 거리 기반으로 유지
- 사용하지 않는 critic, rnn layer, 중복 코드를 정리
- Colab 사용 manual 개선
- bio reward scale tuning

### 2026-02-24

- SAC/RSAC 기반 default 실행 설정 정리
- cast penalty, bio reward scale, total episode, learning rate 기본값 조정
- connectome actor network 추가
- connectome 모델 실행 명령을 Colab notebook에 추가
- default agent를 RSAC 흐름에 맞춰 정리

### 2026-03-04

- GIF 생성 시 평가 episode 중 reward가 가장 높은 seed를 선택하도록 변경
- connectome2 구조 추가
- connectome hidden size 기본값 조정
- manual 및 README 업데이트

### 2026-03-27

- 사용하지 않는 experiment script 제거
- GRU framework notebook 최적화 및 통합
- connectome framework notebook 추가
- connectome metadata 수정

### 2026-04-01

- notebook 위치를 `ipynb` 하위 구조로 정리
- GRU V0/V1, connectome V0/V1 notebook 정리
- notebook output 제거
- MLP model notebook 추가
- complex environment 평가 스크립트 추가

### 2026-05-11

- PPO, RSAC, DRQN, MLP, GRU, connectome을 포함하는 통합 framework로 개편
- agent, env, model, config, factory, plotter, seed 구조 재정리
- PPO agent와 metrics callback 추가
- OSL 2D environment를 모듈화
- 이전 실험 demo를 현재 repository로 통합
- connectome metadata/weights 자산 추가
- DRQN, PPO, RSAC notebook framework 정리

### 2026-05-12

- OSL reward 계산 수정
- PPO notebook과 agent에 TensorBoard logging 추가

### 2026-05-13

- 농도에 대한 linear dense reward 추가
- white noise 기반 field를 Gaussian local noise 기반 field로 변경
- spawn 위치를 odor field에 반영
- curriculum field 시각화 스크립트 추가
- PPO cast log 추가
- SAC agent 및 SAC notebook 추가
- 시작 위치 범위를 더 멀리 확장
- 평가 환경 시각화 버그 수정
- SAC notebook의 local 실행 cell 추가

### 2026-05-14

- OSL 중간 결과와 분석 방향 정리
- recurrent policy 분석 프레임워크 구축
- RPPO 및 CfC 계열 분석 시도
- CartPole actor-critic 분석, 결과 plotting 도구 추가

### 2026-05-30

- OSL2D 분석 도구 도입
- `analysis/osl2d`에 label, latent visualization, probe, neuron, Jacobian, fixed point, ablation 분석 모듈 추가
- PPO-GRU, SAC-GRU notebook framework 추가
- baseline chemotaxis notebook 추가
- chemotaxis baseline 모델 3종 구성
- GRU backbone, connectome model, policy adapter 정리

### 2026-05-31

- baseline chemotaxis 모델 tuning
- sensor 위치를 head 방향으로 이동하고 forward offset 조정
- head cast reward cost와 stopped multiplier 조정
- entropy 및 training log 개선
- radius와 reward scale 조정
- 시간차 정보를 observation에 포함
- curriculum phase 구성과 순서 조정
- PPO/SAC 학습 중단 후 재개 기능 추가
- connectome 구조 변경 및 PPO/SAC와 adapter 반영
- real brain SNN, valence connectome, evolutionary strategy 실험 구조 구성
- noise sweep 분석 notebook 및 script 추가
- valence connectome Colab notebook 구성
- ground truth 설정 수정

### 2026-06-01

- 분석 및 baseline 시각화 개선
- minimal baseline chemotaxis notebook 추가
- presentation asset 생성 notebook 구성
- baseline noise sweep 결과 json과 plotting utility 추가
- trajectory, baseline, active sensing 관련 그림 생성 흐름 정리

### 2026-06-06

- PPO-GRU recurrent critic 실험 추가
- recurrent/stateless critic 선택 구조 개선
- PPO-GRU active sensing top-k PCA 분석 스크립트 추가
- PPO-GRU noise sweep 평가 스크립트 추가

### 2026-06-07

- repository 구조 정리
- `Analysis`를 `analysis`로 정리
- obsolete notebook과 중복 script 제거
- demo, notebook, src README 정리
- 현재 유지할 notebook을 `notebooks` 하위로 정리
- requirements 정리

### 2026-06-09

- stateless critic 구조 복구 및 설정화
- policy adapter, PPO agent, model policy, config, factory 수정
- poster용 figure 생성을 위한 notebook 정리
- 임시 poster scaffold와 plan 파일 제거
