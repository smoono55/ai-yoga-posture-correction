#!/usr/bin/env python3
"""
main.py — Entry point for the Yoga Posture Correction System.

Usage
-----
    python main.py                   # run with live debug window
    python main.py --no-debug        # headless (background) mode
    python main.py --debug           # verbose terminal logging
    python main.py --monitor 2       # capture monitor 2
    python main.py --webcam 1        # use webcam index 1
    python main.py --tolerance 25    # angle tolerance in degrees
    python main.py --cooldown 8      # seconds between audio prompts
"""

from __future__ import annotations
import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Based Background Yoga Posture Correction System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--debug",     action="store_true", help="Verbose logging.")
    parser.add_argument("--no-debug",  action="store_true", help="Hide webcam debug window.")
    parser.add_argument("--monitor",   type=int,   default=None, help="Screen monitor index.")
    parser.add_argument("--webcam",    type=int,   default=None, help="Webcam device index.")
    parser.add_argument("--tolerance", type=float, default=None, help="Joint angle tolerance (°).")
    parser.add_argument("--fps",       type=int,   default=None, help="Processing FPS.")
    parser.add_argument("--cooldown",  type=float, default=None, help="Seconds between prompts.")
    return parser.parse_args()


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    import config
    if args.monitor   is not None: config.SCREEN_MONITOR_INDEX     = args.monitor
    if args.webcam    is not None: config.WEBCAM_INDEX              = args.webcam
    if args.tolerance is not None: config.ANGLE_TOLERANCE_DEGREES   = args.tolerance
    if args.fps       is not None: config.PROCESSING_FPS            = args.fps
    if args.cooldown  is not None: config.FEEDBACK_COOLDOWN_SECONDS = args.cooldown

    from orchestrator import YogaCorrector
    corrector = YogaCorrector(show_debug=not args.no_debug)
    corrector.run()


if __name__ == "__main__":
    main()
