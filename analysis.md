# analysis.md — Connectome Policy Analysis Methodology

생물학적으로 제약된 강화학습 정책 — *Drosophila* larva connectome을 actor backbone으로 가진 RL 정책 — 의 odor source localization 행동을, connectome 내부 기전 중심으로 분석하기 위한 일반 방법론. 환경(2D / 3D)과 학습 알고리즘에 무관하게 적용한다.

## 1. Objective

학습된 connectome 정책이 환경에서 보이는 OSL 행동(run, stop-and-cast, turn 등)을:

- (a) 실측 larva 행동학과 통계적으로 유사한지 (behavioral fidelity)
- (b) 어떤 동역학 모드(slow attractor / oscillatory)에 의해 발현되는지
- (c) 어느 cell type 부분집합이 인과적으로 책임지는지

정량적·기전적으로 입증한다.

## 2. Significance

행동주의 RL 결과("성공률 X%")가 아닌, **신경 회로 수준의 인과 기전**을 보고하는 것이 의의. connectome이 단순 capacity scaffold인지, 실제로 행동을 produce하는 mechanism인지 구분해야 한다.

## 3. Scope

- **분석 대상**: 학습 완료된 connectome 정책 단독. 학습 알고리즘(PPO/SAC/...)이나 환경 차원(2D/3D)은 분석 방법론에 영향을 주지 않는다 — 모든 분석은 actor의 connectome state trace 위에서 수행.
- **Cross-architecture 비교는 본 계획 범위 밖.** 같은 알고리즘으로 학습된 connectome vs non-connectome backbone 비교는 algorithm × architecture confound가 있어 "connectome-specific" 인과 주장에 부적합. mechanistic 분석은 같은 모델 내부 ablation으로 인과를 다룬다.

## 4. Three-Phase Analysis

### Phase 1 — Behavioral Quantification

**목표**: 학습된 정책의 행동 fingerprint를 만들고 실측 larva 데이터와 비교.

**Metric**:

- **Kinematic 시계열 통계**: 속도, body/head 회전속도, head relative angle의 mean / var / autocorrelation. curvature, tortuosity (path_length / displacement).
- **Bearing-projection velocity**: 소스 방향 성분 surge 효율.
- **Cast frequency**: head relative angle 시계열의 PSD 피크. 실측 larva의 1–3 Hz 영역과 비교.
- **Action distribution**: 연속 action 채널 joint histogram.
- **Event ratio**: episode 내 run / cast / turn 비율 (`classify_event` 결과).
- **DTW**: 학습된 정책 궤적 vs 실측 larva 궤적. 별도 dataset 확보 필요.

**조건 sweep**: 환경 난이도(noise 강도) × 다수 eval seed. 난이도별 행동 변화 추적.

**데이터 추출**: eval 시점에 episode 단위로 `(state_trace, info_trace, action_trace, kinematic_trace)`를 jsonl / npz로 저장.

### Phase 2 — Dynamical Systems Analysis

**목표**: connectome hidden state 동역학에서 행동을 설명하는 지배 모드 추출.

**Hidden state**: connectome message-passing의 마지막 iteration 후 state. 외부 env step 단위로 기록.

**Jacobian / Eigenvalue**:

- 외부 step 함수 `f: state[t] ↦ state[t+1]`의 야코비안 `J = ∂f/∂state`.
- `J`는 connectivity sparsity를 따른다 — sparse decomposition 활용.
- **Cell type partition**: layout 메타데이터 (sensory / PN / LN / KC / MBON / ORN / MBON) 기반 grouping. block-wise eigenvalue로 그룹별 dominant mode 분리.
- **Slow mode** (|λ| ≈ 1, real): 농도/방향 정보의 working memory. surge 구간에서 강할 것으로 예상.
- **Oscillatory mode** (복소 λ, |λ| ≈ 1, arg(λ) ≠ 0): cast 진동. cast 구간에서 dominant 예측. `arg(λ)/dt`가 cast 주파수와 일치해야 가설 강화.
- **Eigenvector**: dominant mode의 eigenvector를 cell type별로 투영. 어느 그룹이 mode를 carry하는지 식별.

**Behavior-conditioned analysis**: episode를 행동 라벨로 segment → segment별 J 계산 → 라벨별 spectrum 비교.

### Phase 3 — Mechanistic Interpretability

**목표**: Phase 2에서 식별된 후보 cell type 그룹이 행동에 인과적으로 기여하는지 검증.

**Linear Probing** (정보 표상):

- 입력: 그룹별 state activations.
- 라벨 후보: 소스로부터의 bearing / 거리, sensor asymmetry (`log c_L − log c_R`), `dlog c̄/dt`, behavior event flags, 환경 noise stage (turbulence 인지 여부).
- LinearRegression / LogisticRegression on train/test split. R² / accuracy 보고. 통제: shuffle 라벨 baseline.

**Activation Patching** (인과 검증):

- Phase 2의 dominant mode를 carry하는 cell type group `G`를 식별 → connectome forward에서 `state[:, G_indices]`를 zero / mean / sign-flip 처리.
- 동일 seed에서 ablated 정책의 행동 변화 측정:
  - 행동 event 비율 변화 (cast / run 등)
  - 성공률 변화
  - 궤적의 DTW distance from baseline
- **인과 주장**: "G를 ablate하면 행동 X의 빈도가 N% 떨어지고 성공률 M% 감소" → G가 행동 X의 인과적 기질.

**Patching 단위**:

- Cell type 단위 (빠르고 해석 명확)
- Output 노드 단일 ablation
- 단일 노드 (전체 노드 반복, expensive)

## 5. Infrastructure (Prerequisites)

분석 시작 전 모델/eval 측에 추가 필요:

| 항목 | 목적 |
|---|---|
| Connectome state trace hook | forward step에서 hidden state 누적 — Phase 2/3 입력 |
| Trace dump in eval | episode 단위로 `(state, info, action)` 시계열을 디스크로 |
| Patching hook | forward에 `patch_indices`, `patch_value`(zero/mean/flip) 인자 |
| Jacobian utility | 단일 step Jacobian 계산 + cell type block 분해 + eigendecomp |
| Probing utility | sklearn 래퍼, group mask 기반 linear probe |
| Larva trajectory dataset | Phase 1 DTW blocking 항목 (외부 dataset) |

## 6. Statistical Methodology

- **Replication**: 환경 난이도 × 다수 training seed × 다수 eval seed.
- **Multiple comparison**: metric마다 조건 비교 → Bonferroni / Holm 보정. Patching condition 수도 동일 적용.
- **Effect size**: p-value뿐 아니라 Cliff's delta / Cohen's d.
- **Negative result protocol**: Phase 2 eigenvalue가 명확히 cluster되지 않으면 → "no dominant mode" 보고. Phase 3 patching에서 행동 변화 < 5% → "non-causal" 보고. Falsifiable.
- **Pre-registration**: 분석 시작 전 가설(예: cast 진동 모드 = 특정 그룹 carries) 명시 후 검증.

## 7. Reporting Aid (light)

학습된 episode의 trajectory rendering + Phase 3 patching summary를 multimodal 입력으로 LLM에 보내 자연어 사례 서술 자동화 (figure caption / supplementary narrative). LLM 출력은 정량 결과를 paraphrase하는 역할만. 인과 주장이나 새 발견은 LLM이 만들지 않음.

## 8. Roadmap

1. **Infrastructure**: state trace hook + patching arg + dump 셀. Jacobian / probing utility.
2. **Phase 1**: 학습된 ckpt에서 난이도별 dump → kinematic + event ratio + cast PSD + action distribution. (DTW는 larva dataset 확보 후.)
3. **Phase 2**: dump된 trace에서 Jacobian 시계열 + cell type 단위 eigendecomposition. 행동 segment별 spectrum 비교. dominant mode → cell type 후보.
4. **Phase 3**: 단일 노드 + cell type 단위 patching. seeds × groups × ablation modes.
5. **Reporting**: figure 자동 생성 + (optional) LLM caption.

## 9. Open Questions

- **Inner-step vs outer-step state**: message-passing 반복 중 어느 시점을 hidden state로 볼지. outer-step 마지막이 default지만 inner-step 평균/intermediate도 후보. pilot 후 결정.
- **Sparse Jacobian의 eigenvalue 해석**: connectivity가 강하게 sparse라 일반 RNN spectrum 직관이 안 통할 수 있음. pilot run으로 검증.
- **Stronger ablation**: cell type 단위 patching이 약하면 **shuffled-edge connectome** (같은 노드 수 + 같은 sparsity, edge target만 permute) 학습/비교 고려. same-architecture 비교라 cross-trainer baseline보다 인과적 강함.
- **Cross-environment generalization**: 한 환경(예: 2D)에서 식별된 mechanism이 다른 환경(3D)으로 전이되는지. 같은 분석 protocol 그대로 적용 가능.
- **DTW dataset confound**: larva tracking 실험 조건이 sim과 다를 수 있음. normalize/subset 필요.
