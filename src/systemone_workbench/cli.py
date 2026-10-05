"""Start the System One Workbench server."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve System One decision models (Laya MLX / Core ML, plus registered checkpoints and proxies)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind to this IP; use your Tailscale IP for iPhone access")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument(
        "--vision-checkpoint",
        help="serve the separate local Clef-Flash image demo from this checkpoint directory",
    )
    args = parser.parse_args()
    if args.vision_checkpoint:
        from .vision_demo import create_vision_app

        uvicorn.run(
            create_vision_app(checkpoint=args.vision_checkpoint),
            host=args.host,
            port=args.port,
        )
    else:
        from .app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
