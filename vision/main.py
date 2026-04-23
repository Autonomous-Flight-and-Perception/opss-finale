#!/usr/bin/env python3
"""
OPSS - Optical Projectile Sensing System
Main entry point for the unified vision + physics pipeline.

Usage:
    python main.py                              # Kalman only on :8000
    python main.py --tracker ukf                # CTRV UKF only
    python main.py --tracker pf                 # Particle filter only
    python main.py --tracker adaptive           # Auto-switching KF / UKF / PF
    python main.py --tracker kalman,ukf,pf,adaptive --primary adaptive
                                                # All four in parallel,
                                                #   adaptive drives the cobot
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
        "--tracker", default="kalman",
        help="One or more state estimators (comma-separated). Choices: "
             "'kalman' (linear constant-velocity KF), "
             "'ukf' (CTRV nonlinear UKF, ported from MonteCarlo), "
             "'pf' (3D particle filter, ported from PF repo), "
             "'adaptive' (auto-switches KF/UKF/PF by motion regime). "
             "Pass several for a parallel comparison demo: "
             "--tracker kalman,ukf,pf,adaptive. Default: kalman."
    )
    parser.add_argument(
        "--primary", default=None,
        help="In multi-tracker mode, which tracker drives the cobot wire feed. "
             "Defaults to the first --tracker entry."
    )

    args = parser.parse_args()

    valid_trackers = {"kalman", "ukf", "pf", "adaptive"}
    tracker_names = [t.strip() for t in args.tracker.split(",") if t.strip()]
    unknown = [t for t in tracker_names if t not in valid_trackers]
    if unknown:
        sys.exit(f"[ERROR] Unknown tracker(s): {unknown}. Valid: {sorted(valid_trackers)}")
    if not tracker_names:
        sys.exit("[ERROR] --tracker cannot be empty")

    primary = args.primary or tracker_names[0]
    if primary not in tracker_names:
        sys.exit(f"[ERROR] --primary {primary!r} not in --tracker {tracker_names}")

    print("=" * 60)
    print("  OPSS - Optical Projectile Sensing System v2.0.0")
    print("  Unified Vision + Physics Pipeline")
    print("=" * 60)
    print(f"  Trackers: {', '.join(tracker_names)}")
    if len(tracker_names) > 1:
        print(f"  Primary (drives cobot): {primary}")

    from opss.pipeline.core import OPSSPipeline, PipelineConfig

    config = PipelineConfig(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        tracker_types=tracker_names,
        primary_tracker=primary,
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
