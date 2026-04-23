#!/usr/bin/env python3
"""
OPSS - Optical Projectile Sensing System
Main entry point for the unified vision + physics pipeline.

Usage:
    python main.py                    # Start web server on port 8000
    python main.py --port 8080        # Custom port
    python main.py --no-web           # Run pipeline without web server
    python main.py --tracker ukf_nn   # Use UKF-NN tracker with NN model
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="OPSS - Optical Projectile Sensing System"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--no-web", action="store_true",
        help="Run pipeline without web server"
    )
    parser.add_argument(
        "--capture-width", type=int, default=1280,
        help="Capture resolution width (default: 1280)"
    )
    parser.add_argument(
        "--capture-height", type=int, default=720,
        help="Capture resolution height (default: 720)"
    )
    parser.add_argument(
        "--tracker", choices=["kalman", "ukf_nn"], default="kalman",
        help="Tracker type (default: kalman)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to UKF-NN model weights (.weights.npz or .pt)"
    )
    parser.add_argument(
        "--stats", type=str, default=None,
        help="Path to feature normalization stats (.json)"
    )

    args = parser.parse_args()

    # Auto-set model/stats paths when ukf_nn tracker is selected
    if args.tracker == "ukf_nn":
        if args.model is None:
            args.model = "models/nn_3d.weights.npz"
        if args.stats is None:
            args.stats = "models/feat_stats_3d.json"

    print("=" * 60)
    print("  OPSS - Optical Projectile Sensing System v2.0.0")
    print("  Unified Vision + Physics Pipeline")
    print("=" * 60)
    print(f"  Tracker: {args.tracker}")
    if args.tracker == "ukf_nn":
        print(f"  Model:   {args.model}")
        print(f"  Stats:   {args.stats}")

    from opss.pipeline.core import OPSSPipeline, PipelineConfig

    config = PipelineConfig(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        tracker_type=args.tracker,
        ukf_nn_model_path=args.model,
        ukf_nn_stats_path=args.stats,
    )

    if args.no_web:
        # Run pipeline without web server
        pipeline = OPSSPipeline(config)

        print("\n[OPSS] Starting pipeline (no web server)...")
        if pipeline.start():
            print("[OPSS] Pipeline running. Press Ctrl+C to stop.")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[OPSS] Shutting down...")
                pipeline.stop()
        else:
            print("[OPSS] Failed to start pipeline")
            sys.exit(1)
    else:
        # Run with web server
        print(f"\n[OPSS] Starting web server on http://{args.host}:{args.port}")
        print("[OPSS] Dashboard: http://localhost:{}/".format(args.port))
        print("[OPSS] API docs: http://localhost:{}/docs".format(args.port))
        print()

        # Create pipeline singleton with our config before web app uses it
        from opss.pipeline.core import get_pipeline
        get_pipeline(config)

        from opss.web.app import app
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
