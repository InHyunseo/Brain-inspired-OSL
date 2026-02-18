import os
import time
import argparse
import torch

# 우리가 만든 모듈들 가져오기
from train import train
from eval import evaluate

def main():
    # ==========================================
    # 1. 통합 설정 (Configuration)
    # ==========================================
    parser = argparse.ArgumentParser(description="End-to-End RL Pipeline")
    
    # [공통]
    parser.add_argument("--env-id", default="OdorHold-v3")
    parser.add_argument("--out-dir", default="runs", help="Root directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")

    # [Train 관련]
    parser.add_argument("--agent-type", choices=["drqn", "dqn"], default="drqn")
    parser.add_argument("--total-episodes", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rnn-hidden", type=int, default=147)
    parser.add_argument("--dqn-hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    
    # Hyperparams (Train.py와 변수명 일치시킴)
    parser.add_argument("--buffer-size", type=int, default=150000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--target-update-every", type=int, default=20)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=20)
    
    # Env Params
    parser.add_argument("--src-x", type=float, default=0.0)
    parser.add_argument("--src-y", type=float, default=0.0)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--sigma-c", type=float, default=1.0)

    # [Eval/Plot 관련]
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout GIF during evaluation (default: True)",
    )

    args = parser.parse_args()

    # 실행 이름 자동 생성 (예: drqn_main_20260215_1200)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.agent_type}_main_{timestamp}"
    
    # args 객체에 run_name 주입 (train.py가 사용)
    args.run_name = run_name
    
    # ==========================================
    # 2. 학습 (TRAIN)
    # ==========================================
    print(f"\n{'='*40}")
    print(f"🚀 [Step 1] Training Started: {run_name}")
    print(f"{'='*40}")
    
    try:
        # train.py의 핵심 함수 호출
        train(args)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Proceeding to evaluation...")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # ==========================================
    # 3. 경로 연결 (Bridge)
    # ==========================================
    # train.py가 생성했을 경로를 계산
    run_dir = os.path.join(args.out_dir, run_name)
    
    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory not found at {run_dir}")
        return

    print(f"\n✅ Training Finished. Results saved at: {run_dir}")

    # ==========================================
    # 4. 평가 및 시각화 (EVAL & PLOT)
    # ==========================================
    print(f"\n{'='*40}")
    print(f"📊 [Step 2] Evaluation & Plotting")
    print(f"{'='*40}")

    # eval.py를 위한 인자 설정 주입
    # (eval.py는 args.run_dir와 args.episodes를 필요로 함)
    args.run_dir = run_dir
    args.episodes = args.eval_episodes
    args.ckpt = None # Auto-detect best.pt

    try:
        evaluate(args)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return

    print(f"\n{'='*40}")
    print(f"🎉 All Jobs Done! Check '{run_dir}' for plots.")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
