"""Start the local decision server."""

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Laya-MLX through the System One API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind to this IP; use your Tailscale IP for iPhone access")
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
