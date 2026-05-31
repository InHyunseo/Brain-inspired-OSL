# Analysis 파이프라인 — 수식 정리

학습된 connectome 정책의 OSL 행동을 **신경 회로 수준의 인과 기전**으로 설명하기 위한 분석 파이프라인의 각 단계 수식 정리.
전체 방법론은 [`../analysis.md`](../analysis.md), 구현은 [`osl2d/`](osl2d/), 실행 순서는 [`osl2d/run_all.py`](osl2d/run_all.py) (collect → P1 → P2a → P2b → P2c → P3a → P3b → P4).

> 모든 분석의 공통 입력은 eval rollout에서 덤프한 시계열
> $(o_t,\, h_t,\, a_t,\, \text{event flags},\, \text{success})$ 하나다. 전부 hidden state trace 위에서 계산된다.

---

## 표기

| 기호 | 의미 |
|---|---|
| $t$ | 외부 env step (에피소드 $e$, 라벨 $\ell_t$) |
| $o_t \in \mathbb{R}^{d_o}$ | observation |
| $h_t \in \mathbb{R}^{N}$ | hidden (connectome) state |
| $a_t$ | action |
| $F:(o_t,h_t)\mapsto(\mu_t,\log\sigma_t,h_{t+1})$ | 정책의 외부 step map |
| $f:(o_t,h_t)\mapsto h_{t+1}$ | hidden 동역학만 분리한 map |
| $\Delta t = 0.1\,\mathrm{s}$ | env step 시간 ([`phase1_label.py`](osl2d/phase1_label.py) `_DT`) |
| $K = 5$ | 라벨 수, $N$ = hidden 차원, $T$ = 총 step 수 |

라벨 집합: $\ell_t \in \{\mathrm{RUN},\ \mathrm{CAST},\ \mathrm{TURN},\ \mathrm{SPIN},\ \mathrm{STOP}\}$.

---

## 0. 라벨링 — 모든 분석의 좌표축

[`osl2d/segment.py`](osl2d/segment.py). 6개 boolean event flag를 우선순위로 하나의 라벨로 collapse (우선순위 SPIN > CAST > TURN > RUN > STOP):

$$
\ell_t=\begin{cases}
\mathrm{SPIN} & b^{\mathrm{spin}}_t\\[2pt]
\mathrm{CAST} & b^{\mathrm{lsweep}}_t \lor b^{\mathrm{hcast}}_t\\[2pt]
\mathrm{TURN} & b^{\mathrm{turn}}_t \land \lnot\, b^{\mathrm{run}}_t\\[2pt]
\mathrm{RUN}  & b^{\mathrm{run}}_t\\[2pt]
\mathrm{STOP} & \text{otherwise}
\end{cases}
$$

이후 모든 phase가 "이 라벨일 때 vs 저 라벨일 때"로 hidden state를 쪼개 비교한다. 라벨이 분석의 축이다.

---

## Phase 1 — 행동 정량화

[`osl2d/phase1_label.py`](osl2d/phase1_label.py)

**라벨 비율**

$$
\rho_i = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\ell_t = i]
$$

**전이행렬** — 인접 step 라벨쌍 카운트 후 행 정규화

$$
C_{ij}=\sum_{t=1}^{T-1}\mathbf{1}[\ell_t=i]\,\mathbf{1}[\ell_{t+1}=j],
\qquad
\widehat{M}_{ij}=\frac{C_{ij}}{\max\!\left(1,\ \sum_k C_{ik}\right)}
$$

**Cast PSD** — CAST 구간의 head-relative-angle $\psi_t$만 모아 평균 제거 후 Welch 추정

$$
x_t = \psi_t - \bar\psi \quad (\ell_t=\mathrm{CAST}),
\qquad
\widehat P(f)=\mathrm{Welch}\!\left(x;\ f_s = 1/\Delta t\right),
\qquad
f^\star = \arg\max_f \widehat P(f)
$$

피크 주파수 $f^\star$를 실측 larva의 1–3 Hz와 비교 (행동 충실도). nperseg $=\min(128, |x|)$, 표본 < 16이면 skip.

---

## Phase 2a — 표상 분리도

[`osl2d/phase2a_latent_viz.py`](osl2d/phase2a_latent_viz.py). hidden state $H \in \mathbb{R}^{T \times N}$ 를 차원축소:

$$
Z = \mathrm{PCA}_{k}(H),\quad k=\min(50, N)
\qquad\xrightarrow{\ \mathrm{UMAP}\ }\qquad
Z_2 \in \mathbb{R}^{T \times 2}
$$

라벨 $\ell$ 기준 $Z_2$ 위에서 분리도 두 지표.

**Silhouette** — 점 $i$에 대해 $a_i$ = 같은 라벨 내 평균거리, $b_i$ = 가장 가까운 타 라벨 평균거리:

$$
s_i=\frac{b_i-a_i}{\max(a_i,\,b_i)},
\qquad
S=\frac{1}{T}\sum_{i} s_i \ \in [-1,\,1]
$$

**Calinski–Harabasz** — 군집간 산포 $B_K$, 군집내 산포 $W_K$:

$$
\mathrm{CH}=\frac{\operatorname{tr}(B_K)}{\operatorname{tr}(W_K)}\cdot\frac{T-K}{K-1}
$$

둘 다 클수록 라벨이 latent 공간에서 잘 분리됨 = hidden state가 행동을 선형 분리 가능한 형태로 담음.

---

## Phase 2b — 선형 probe

[`osl2d/phase2b_probe.py`](osl2d/phase2b_probe.py), [`osl2d/probe.py`](osl2d/probe.py).
입력은 hidden state 전체 $h_t$, 타깃은 라벨 $\ell_t$. Multinomial logistic regression:

$$
P(\ell=c \mid h)=\frac{\exp(w_c^\top h + b_c)}{\sum_{c'}\exp(w_{c'}^\top h + b_{c'})},
\qquad
\hat\ell = \arg\max_c P(\ell=c \mid h)
$$

**Episode-level split**: 에피소드 집합을 셔플해 20%를 test로. 같은 에피소드가 train/test에 걸치지 않음 → 인접 프레임 누수 차단.

지표 (test 집합 $\mathcal{T}$ 위에서):

$$
\mathrm{acc}=\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}\mathbf{1}[\hat\ell_t=\ell_t],
\qquad
\text{macro-F1}=\frac{1}{K}\sum_{c}\frac{2\, P_c R_c}{P_c+R_c}
$$

**Shuffle baseline**: train 라벨을 permute한 $\tilde\ell$로 학습한 모델의 test acc $\mathrm{acc}_{\mathrm{shuf}}$.
$\mathrm{acc} \gg \mathrm{acc}_{\mathrm{shuf}}$ 여야 정보가 실재.

---

## Phase 2c — 뉴런별 기여도

[`osl2d/phase2c_neuron.py`](osl2d/phase2c_neuron.py). probe 가중치 행렬 $W \in \mathbb{R}^{K \times N}$ 의 절댓값을 기여도로:

$$
\mathrm{contrib}[c,n]=\lvert W_{c,n}\rvert
$$

라벨 $c$의 top-$k$ 뉴런 집합 (이게 `neuron_groups.json` → Phase 4 ablation 타깃):

$$
G_c=\operatorname*{top\text{-}k}_{n}\,\bigl(\lvert W_{c,n}\rvert\bigr)
$$

cell-type 그룹 $A$ 와의 overlap:

$$
\text{frac-of-group}=\frac{\lvert G_c \cap A\rvert}{\lvert A\rvert},
\qquad
\text{frac-of-top}=\frac{\lvert G_c \cap A\rvert}{\lvert G_c\rvert}
$$

---

## Phase 3a — Jacobian eigenmode

[`osl2d/jacobian.py`](osl2d/jacobian.py), [`osl2d/phase3a_jacobian.py`](osl2d/phase3a_jacobian.py).
각 관측점에서 hidden 동역학의 국소 선형화 (autograd):

$$
J_t=\left.\frac{\partial f(o_t,\,h)}{\partial h}\right|_{h=h_t} \in \mathbb{R}^{N\times N},
\qquad
\{\lambda_k\}=\operatorname{eig}(J_t)
$$

지배 고윳값 $\lambda^\star=\lambda_{\arg\max_k |\lambda_k|}$ 에 대해:

$$
\lvert\lambda^\star\rvert,
\qquad
\arg(\lambda^\star),
\qquad
\text{oscillatory} \iff \lvert\operatorname{Im}\lambda^\star\rvert > 10^{-6}
$$

모드 카운트 (`#` = 개수):

$$
n_{\mathrm{slow}}=\#\bigl\{k:\ \lvert\operatorname{Im}\lambda_k\rvert<10^{-6}\ \land\ \lvert\lambda_k\rvert>0.9\bigr\}
$$

$$
n_{\mathrm{osc}}=\tfrac{1}{2}\,\#\bigl\{k:\ \lvert\operatorname{Im}\lambda_k\rvert>10^{-6}\bigr\}
$$

라벨별 $S$개 샘플 통계: $\overline{\lvert\lambda^\star\rvert}_\ell,\ \lvert\lambda^\star\rvert^{(p90)}_\ell,\ \overline{\arg\lambda^\star}_\ell,\ \mathrm{osc\text{-}frac}_\ell,\ \mathrm{slow\text{-}frac}_\ell$.

**해석**

- **Slow mode**: $\lambda \in \mathbb{R},\ \lvert\lambda\rvert\approx 1$ → 정보 적분 / working memory (surge 구간에서 강할 것).
- **Oscillatory mode**: $\lambda$ 복소, $\lvert\lambda\rvert\approx 1$ → 진동. 추정 주파수

$$
f_{\mathrm{osc}}=\frac{\lvert\arg\lambda^\star\rvert}{2\pi\,\Delta t}
$$

가 Phase 1의 $f^\star$ 와 일치하면 "CAST = 이 진동 모드" 가설 강화.

**Block 분해**: cell-type 인덱스 집합 $A$ 로 부분행렬 $J[A,A]$ 의 고윳값 → 그룹별 dominant mode.

> Caveat: $J_t$ 는 관측점 $h_t$ 주변 국소 선형화라 비선형 영역은 못 본다.

---

## Phase 3b — fixed / slow point

[`osl2d/phase3b_fixedpoint.py`](osl2d/phase3b_fixedpoint.py).
라벨별 대표 입력 $\bar o_\ell$ 고정, $\bar h_\ell + \epsilon$ 에서 L-BFGS로 고정점 탐색:

$$
h^\star=\arg\min_h\ \bigl\lVert f(\bar o_\ell,\,h)-h \bigr\rVert^2,
\qquad
\mathrm{residual}=\bigl\lVert f(\bar o_\ell,\,h^\star)-h^\star \bigr\rVert^2
$$

$h^\star$ 의 Jacobian eigen으로 안정성 분류:

$$
\lvert\lambda^\star(h^\star)\rvert
\begin{cases}
\approx 1 & \text{느린 attractor (메모리 구조)}\\
< 1 & \text{수렴 (안정)}\\
> 1 & \text{발산}
\end{cases}
$$

plot은 라벨별 best point (residual 최소)의 $\lvert\lambda^\star\rvert$ 를 $\lvert\lambda\rvert=1$ 기준선과 비교.

---

## Phase 4 — 인과 ablation

[`osl2d/phase4_ablation.py`](osl2d/phase4_ablation.py).
타깃 그룹 $G$ 를 매 step zero-patch ($h[G]\leftarrow 0$). 같은 seed로 baseline과 비교.

라벨 분포 $p^{\mathrm{abl}},\ p^{\mathrm{base}}$ 에 대해:

$$
\Delta\mathrm{return}=\bar R^{\mathrm{abl}}-\bar R^{\mathrm{base}},
\qquad
\Delta\mathrm{succ}=\overline{\mathrm{succ}}^{\,\mathrm{abl}}-\overline{\mathrm{succ}}^{\,\mathrm{base}}
$$

$$
D_{\mathrm{KL}}\!\left(p^{\mathrm{abl}} \,\big\|\, p^{\mathrm{base}}\right)=\sum_{\ell} p^{\mathrm{abl}}_\ell \,\log\frac{p^{\mathrm{abl}}_\ell}{p^{\mathrm{base}}_\ell}
\qquad (\text{floor } \epsilon=10^{-6})
$$

$$
\Delta p_\ell = p^{\mathrm{abl}}_\ell - p^{\mathrm{base}}_\ell
$$

**인과 주장**: $G$ ablate 시 $\lvert\Delta p_{\mathrm{CAST}}\rvert$ 가 크고 $\Delta\mathrm{succ}<0$ → "$G$ 가 행동 X의 인과적 기질".
[`../analysis.md`](../analysis.md) §6 사전등록 기준상 행동 변화 $< 5\%$ 면 **non-causal** 로 falsifiable 보고.

---

## 전체 논리 사슬

$$
\underbrace{\rho,\ \widehat M,\ f^\star}_{\text{P1: 무슨 행동인가}}
\ \longrightarrow\
\underbrace{S,\ \mathrm{CH},\ \mathrm{acc}\gg\mathrm{acc}_{\mathrm{shuf}}}_{\text{P2: hidden이 행동을 담는가 (상관)}}
\ \longrightarrow\
\underbrace{\lvert\lambda^\star\rvert,\ \arg\lambda^\star,\ h^\star}_{\text{P3: 어떤 동역학 모드인가}}
\ \longrightarrow\
\underbrace{\Delta\mathrm{succ},\ D_{\mathrm{KL}}}_{\text{P4: 그 뉴런이 인과적인가}}
$$

P2까지는 상관 / 디코더빌리티, **P3·P4가 "connectome이 capacity scaffold가 아니라 실제 mechanism"임을 보이는 인과 핵심**이다.
