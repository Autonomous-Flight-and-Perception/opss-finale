#!/usr/bin/env python3
"""
OPSS - Optical Projectile Sensing System
Main entry point for the unified vision + physics pipeline.

Usage:
    python main.py                    # Start web server on port 8000
    python main.py --port 8080        # Custom port
    python main.py --no-web           # Run pipeline without web server
    python main.py --tracker ukf       # Use Unscented Kalman Filter
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
        "--tracker", choices=["kalman", "ukf"], default="kalman",
        help="State estimator: 'kalman' (linear KF, pixel-space) or "
             "'ukf' (Unscented KF, meter-space, gravity in process model). "
             "Default: kalman."
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  OPSS - Optical Projectile Sensing System v2.0.0")
    print("  Unified Vision + Physics Pipeline")
    print("=" * 60)
    print(f"  Tracker: {args.tracker}")

    from opss.pipeline.core import OPSSPipeline, PipelineConfig

    config = PipelineConfig(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        tracker_type=args.tracker,
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
