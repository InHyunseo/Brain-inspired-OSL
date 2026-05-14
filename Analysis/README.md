# Analysis — Methodology Dry-Run (NCP × POMDP Pendulum)

`../analysis.md`에 정의된 3-phase 분석 방법론을 **실제 connectome 학습에 투입하기 전에** 상용 오픈소스 모델 위에서 검증하기 위한 subproject.

## 무엇을, 왜

- **Surrogate 모델**: Neural Circuit Policies — CfC cell + AutoNCP wiring. AutoNCP는 sensory / inter / command / motor 4-layer 구조로 unit을 자동 분배하고 각 그룹 인덱스를 노출한다. analysis.md §4 Phase 2의 cell-type partition을 그대로 시험 가능.
- **Surrogate task**: `Pendulum-v1`을 velocity-masked POMDP로 (`obs = [cos θ, sin θ]`만). 정책이 hidden state에 속도를 적분해야 풀리며, swing-up ↔ balance 두 행동 모드가 명확해 segment-conditioned spectrum 비교의 자연 가설이 박힘.
- **목적**: phase1~3 분석 코드가 task/policy-agnostic하게 동작하는지 검증. 통과 시 `ncp_policy.py` + `pendulum_pomdp.py`만 connectome/OSL 버전으로 교체하면 phase1~3 그대로 재사용.

## 사전등록 성공 기준

1. **Dynamics**: swing-up vs balance segment 간 dominant eigenvalue cluster가 통계적으로 다르다 (Cliff's δ > 0.3, n_seed ≥ 5).
2. **Causality**: command 그룹 ablation 시 balance 유지율 ≥ 5% 하락.
3. **Representation**: inter 그룹 activation → 가려진 angular velocity linear probe R² > 0.5.

3개 중 ≥ 2개 통과면 OSL connectome 이행 GO. 모두 미달이면 REPORT.md에 진단 후 재설계.

## 실행

```bash
pip install -r requirements.txt
python -m Analysis.train --seed 0 --timesteps 200000
python -m Analysis.eval_dump --run_id ncp_pendulum --seeds 0 1 2 3 4 --episodes 20
python -m Analysis.phase1_behavior --run_id ncp_pendulum
python -m Analysis.phase2_dynamics --run_id ncp_pendulum
python -m Analysis.phase3a_probe   --run_id ncp_pendulum
python -m Analysis.phase3b_patch   --run_id ncp_pendulum
```

## 디렉토리

- `ncp_policy.py` — analysis.md §5의 prerequisite 인터페이스(`forward(obs,h,patch=None)` + `group_indices`) 표준 정의처.
- `pendulum_pomdp.py` — velocity-masked Pendulum wrapper.
- `train.py` / `eval_dump.py` — ckpt + episode trace (`runs/{run_id}/...`).
- `phase{1,2,3a,3b}_*.py` — 분석 스크립트. 모두 `runs/*/eval_*.npz` 만 소비.
- `utils/{segment,jacobian,probe}.py` — phase 공통 헬퍼.

## 라이선스 / 인용

NCP (`ncps`)는 Apache-2.0이며 본 폴더는 의존성으로 사용한다. `NOTICE` 참조.

논문 인용:
- Lechner, M. et al. "Neural circuit policies enabling auditable autonomy." *Nature Machine Intelligence* 2 (2020): 642–652.
- Hasani, R. et al. "Closed-form continuous-time neural networks." *Nature Machine Intelligence* 4 (2022): 992–1003.
