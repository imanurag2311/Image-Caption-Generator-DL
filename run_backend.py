"""
run_backend.py
==============
Entry-point script to start the Flask backend API for the
Image Caption Generator.

Usage
-----
    python run_backend.py                     # defaults: 0.0.0.0:5000
    python run_backend.py --port 8080         # custom port
    python run_backend.py --host 127.0.0.1    # localhost only
    python run_backend.py --debug             # Flask debug mode

Environment variable overrides also work:
    FLASK_HOST=127.0.0.1  FLASK_PORT=8080  python run_backend.py
"""

import argparse
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Start the Image Caption Generator Flask API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("FLASK_HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_PORT", "5000")),
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes"),
        help="Enable Flask debug mode",
    )
    args = parser.parse_args()

    # Import here so argparse errors show before heavy TF imports
    from backend.server import app, logger

    logger.info("=" * 60)
    logger.info("  Image Caption Generator — Backend API")
    logger.info("=" * 60)
    logger.info("  Host  : %s", args.host)
    logger.info("  Port  : %d", args.port)
    logger.info("  Debug : %s", args.debug)
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
