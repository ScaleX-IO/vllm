#!/usr/bin/env python3
import argparse

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response


def create_app(prefill_url: str, decode_url: str, timeout: float) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        payload = await request.json()
        prefill_payload = dict(payload)
        prefill_payload.update(max_tokens=1, stream=False)
        async with httpx.AsyncClient(timeout=timeout) as client:
            prefill = await client.post(
                f"{prefill_url}/v1/completions", json=prefill_payload
            )
            prefill.raise_for_status()
            decode = await client.post(f"{decode_url}/v1/completions", json=payload)
        return Response(
            content=decode.content,
            status_code=decode.status_code,
            media_type=decode.headers.get("content-type"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.prefill_url, args.decode_url, args.timeout),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
