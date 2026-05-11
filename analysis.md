# analysis.md — 2D OSL Connectome Policy Analysis Plan

생물학적으로 제약된 강화학습 정책(larva connectome 기반 PPO)이 2D OSL 환경에서 보이는 행동을 connectome 내부 기전 중심으로 분석하기 위한 실행 계획.

## 1. Research Objective

387-노드 larva connectome(`src/models/connectome.py`)을 actor backbone으로 가진 PPO 정책이, 양측 stereo sensor + 독립 head/body 회전축 환경(`src/envs/osl_env.py`)에서 보이는 OSL 행동(stop-and-cast, surge)이

- (a) 실측 larva 행동학과 통계적으로 유사한지 (behavioral fidelity)
- (b) 어떤 동역학적 모드 (slow attractor / 진동 모드) 에 의해 발현되는지
- (c) 어느 cell type 부분집합 (left/right ORN, PN, LN, KC, MBON) 이 인과적으로 책임지는지

를 정량적·기전적으로 입증한다.

## 2. Scope

- **분석 대상**: PPO로 학습된 connectome 정책 단독.
- **Cross-architecture 비교는 본 계획 범위 밖.** PPO 메인 트랙엔 GRU/MLP backbone이 없고, RSAC backbone들과 비교는 학습 알고리즘 confound가 커서 "connectome-specific" 인과 주장에 부적합.
- 핵심 contribution은 **connectome 내부 mechanism**: 어떤 cell group이 어떤 동역학 모드를 carry하고, 그 모드가 어떤 행동에 인과적으로 기여하는지.

## 3. Behavior Labels

`OslEnv.step` 반환 info dict의 `event_is_*` 키를 행동 라벨로 사용 (`src/envs/events.py::classify_event` 결과).

| 라벨 | 정의 |
|---|---|
| `event_is_run` | `v > run_threshold` + `|body_omega| < 12°/s` |
| `event_is_high_cast_like` | `v < stop_threshold` + 누적 head sweep `37°–120°` |
| `event_is_low_sweep` | `v < stop_threshold` + 누적 head sweep `10°–37°` |
| `event_is_turn_like` | `|body_omega| > 12°/s` 또는 sweep ≥ 20° |
| `event_is_spin_like` | sweep > 120° 또는 sampling 지속 > 2s (페널티 발동) |

추가 metric:
- **Bearing-projection velocity**: `v_mm_s × cos(bearing_to_source_rad)` (info dict)
- **Plume gradient**: `gradient_x`, `gradient_y` (info dict)
- **Sensor asymmetry**: `b_log = log(c_L+ε) − log(c_R+ε)`

## 4. Three-Phase Analysis

### Phase 1 — Behavioral Quantification

**목표**: connectome PPO 정책의 행동 fingerprint를 만들고 실측 larva 데이터와 비교.

**Metric**:
- **Kinematic 시계열 통계**: `v_mm_s`, `body_omega_rad_s`, `head_omega_rad_s`, `head_relative_angle_rad`의 mean / var / autocorrelation. curvature = `body_omega / v`, tortuosity = path_length / displacement.
- **Bearing-projection velocity**: surge 효율.
- **Cast frequency**: `head_relative_angle_rad` 시계열의 PSD 피크 (scipy `welch`). 실측 larva의 1–3 Hz 영역과 비교.
- **Action distribution**: `(v, body_omega, head_omega)` joint histogram.
- **Event ratio**: episode 내 `event_is_run` / `event_is_high_cast_like` step 비율.
- **DTW**: 궤적 `(x_mm, y_mm)` vs 실측 larva 궤적. **요건**: larva tracking dataset 확보 (Gomez-Marin/Louis, Demir/Emonet 류, TODO).

**조건 sweep**: noise_stage (0/1/2) × strength × 30 eval seed. 환경 난이도별 행동 변화 추적.

**데이터 추출**: eval 노트북 셀에서 episode당 info dict + `(env.x_mm, env.y_mm, env.heading_rad, env.head_relative_angle_rad)`를 jsonl로 dump.

### Phase 2 — Dynamical Systems Analysis

**목표**: 423-노드 connectome 활성 동역학에서 행동을 설명하는 지배 모드 추출.

**Hidden state**: `Connectome.forward_step` 6 message-passing iterations 중 마지막 iteration 후의 state `(B, 423)`. outer env step 단위로 기록.

**Jacobian / Eigenvalue**:
- 한 outer step 함수 `f: state[t] ↦ state[t+1]`의 야코비안 `J = ∂f/∂state` (`torch.autograd.functional.jacobian`).
- `J`는 423×423이지만 `edge_targets` 외 entry는 0 — sparse 구조.
- **Cell type partition**: `ConnectomeLayout.{left_orn_indices(21), right_orn_indices(21), mbon_indices(48)}` + metadata csv의 `celltype` (sensory/PN/LN/KC/MBON)으로 grouping. block-wise eigenvalue로 그룹별 dominant mode 분리.
- **Slow mode** (|λ| ≈ 1, real): 농도/방향 정보의 working memory. `event_is_run` 구간에서 강할 것으로 예상.
- **Oscillatory mode** (복소 λ, |λ| ≈ 1, arg(λ) ≠ 0): cast 진동. `event_is_high_cast_like` 구간에서 dominant 예측. `arg(λ)/dt`가 cast 주파수와 일치해야 가설 강화.
- **Eigenvector**: 각 dominant mode의 eigenvector를 cell type별로 투영. 어느 그룹이 mode를 carry하는지 (예: KC가 slow mode, LN이 oscillatory mode).

**Behavior-conditioned analysis**: episode를 `event_is_*` 라벨로 segment → segment별 J 계산 → 라벨별 spectrum 비교.

### Phase 3 — Mechanistic Interpretability

**목표**: Phase 2 후보 cell type 그룹이 행동에 인과적으로 기여하는지 검증.

**Linear Probing** (정보 표상):
- 입력: `state[:, group_indices]` (B*T × |group|).
- 라벨 후보 (info dict):
  - `bearing_to_source_rad` (회귀)
  - `distance_to_source_mm` (회귀)
  - `b_log` (회귀)
  - `dlog_avg/dt` (회귀)
  - `event_is_high_cast_like` (분류)
  - `noise_stage` (분류; turbulence 인지 여부)
- LinearRegression / LogisticRegression on 80/20 split. R² / accuracy 보고. 통제: shuffle 라벨 baseline.

**Activation Patching** (인과 검증):
- Phase 2에서 dominant mode를 carry하는 cell type group `G`를 식별 → `Connectome.forward_step` 내부에서 `state[:, G_indices]`를 zero / mean / sign-flip 처리.
- 동일 seed에서 ablated 정책의 행동 변화 측정:
  - `event_is_high_cast_like` 비율 변화
  - 성공률 변화
  - 궤적의 DTW distance from 원본
- **인과 주장**: "G를 ablate하면 cast 빈도가 X% 떨어지고 success rate가 Y% 감소" → G가 cast 행동의 인과적 기질.

**Patching 단위**:
- Cell type 단위: left_orn / right_orn / PN / LN / KC / MBON (6 groups, 빠르고 해석 명확)
- Output 노드 32개 단일 ablation
- 단일 노드 (423번 반복, expensive)

## 5. Infrastructure

분석 시작 전 추가 필요:

| 항목 | 위치 | 작업 |
|---|---|---|
| Connectome state trace | `src/models/connectome.py` | `forward_step`에 optional `state_history` 누적 또는 forward hook helper |
| Trace dump in eval | eval 노트북 | episode 단위로 `(state_trace, info_trace, action_trace)`를 jsonl/npz로 저장 |
| Patching hook | `src/models/connectome.py` | `forward_step(... patch_indices=None, patch_value=0.0)` 인자 추가, zero/mean/flip 모드 |
| Jacobian utility | `src/utils/dynamics.py` (신규) | `compute_jacobian(connectome, state, sensor_obs)` + `eigendecompose_by_celltype(J, layout)` |
| Probing utility | `src/utils/probing.py` (신규) | `linear_probe(activations, labels, group_mask)` (sklearn 래퍼) |
| Larva trajectory dataset | `assets/larva_tracking/` (신규) | 외부 dataset 다운로드 + parser. Phase 1 DTW만 blocking |

## 6. Statistical Methodology

- **Replication**: noise_stage × 5 training seed × 30 eval seed.
- **Multiple comparison**: Phase 1 metric마다 noise_stage 조건 비교 → Bonferroni / Holm 보정. Phase 3 patching 6 group × 3 ablation = 18 condition도 동일.
- **Effect size**: p-value뿐 아니라 Cliff's delta / Cohen's d 보고.
- **Negative result protocol**: Phase 2 eigenvalue가 명확히 cluster되지 않으면 → "no dominant mode" 보고. Phase 3 patching에서 행동 변화 < 5% → "non-causal" 보고. Falsifiable.
- **Pre-registration**: 분석 시작 전 가설 (예: cast 진동 모드 = LN 그룹 carries) 명시 후 검증.

## 7. Discussion: VLM as Reporting Aid (light)

학습된 episode의 GIF (`render_rollout_frame` 결과) + Phase 3 patching summary table을 multimodal 입력으로 GPT-4V/Claude API에 보내, 자연어 사례 서술 자동화 (figure caption / supplementary narrative).

VLM 출력은 정량 결과를 paraphrase하는 역할만. 인과 주장이나 새 발견은 VLM이 만들지 않음. 논문 main claim에는 미언급, supplementary / discussion 정도.

## 8. Roadmap

1. **Infrastructure (1–2주)**: state trace hook + patching arg + dump 노트북 셀. Jacobian/probing utility.
2. **Phase 1 (1주)**: connectome PPO ckpt에서 noise_stage별 dump → kinematic + event ratio + cast PSD + action distribution. (DTW는 dataset 확보 후.)
3. **Phase 2 (2주)**: dump된 trace에서 Jacobian 시계열 + cell type 단위 eigendecomposition. 행동 segment별 spectrum 비교. dominant mode → cell type 후보.
4. **Phase 3 (2주)**: 단일 노드 + cell type 단위 patching. 5 seed × 6 group × 3 ablation mode = 90 condition.
5. **Reporting (1주)**: figure 자동 생성 + (optional) VLM caption.

## 9. Open Questions

- **Inner-step vs outer-step state**: 6회 반복 중 어느 시점을 hidden state로 볼지. outer-step 마지막이 default지만 inner-step 평균/intermediate도 후보. pilot 후 결정.
- **Sparse Jacobian의 eigenvalue 해석**: connectivity mask가 강하게 sparse(10% density)라 일반 RNN 류 spectrum 직관이 안 통할 수 있음. pilot run으로 검증.
- **Stronger ablation**: cell type 단위 patching이 약하면 후속으로 **shuffled-edge connectome** (같은 노드 수 + 같은 sparsity, edge target만 permute) 학습/비교 고려. same-architecture 비교라 cross-trainer baseline보다 인과적 강함.
- **DTW dataset confound**: larva tracking 실험 조건(arena, plume, dt)이 sim과 다를 수 있음. normalize/subset 필요.
