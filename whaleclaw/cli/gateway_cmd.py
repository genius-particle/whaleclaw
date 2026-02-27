"""CLI sub-command: gateway management."""

from __future__ import annotations

import sys

import typer
import uvicorn

from whaleclaw.config.loader import load_config
from whaleclaw.config.paths import ensure_dirs
from whaleclaw.gateway.app import create_app
from whaleclaw.utils.log import setup_logging

gateway_app = typer.Typer()

_UNSET = -1


@gateway_app.command("run")
def run(
    port: int = typer.Option(_UNSET, help="Gateway 端口"),
    bind: str = typer.Option("", help="绑定地址"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
) -> None:
    """启动 WhaleClaw Gateway。"""
    ensure_dirs()

    gw: dict[str, object] = {}
    if port != _UNSET:
        gw["port"] = port
    if bind:
        gw["bind"] = bind
    if verbose:
        gw["verbose"] = True

    cli_overrides: dict[str, object] = {}
    if gw:
        cli_overrides["gateway"] = gw

    config = load_config(cli_overrides=cli_overrides)

    setup_logging(verbose=config.gateway.verbose)

    app = create_app(config)

    p = config.gateway.port
    b = config.gateway.bind

    sys.stdout.write(f"\n  WhaleClaw Gateway → http://{b}:{p}\n\n")
    sys.stdout.flush()

    uvicorn.run(
        app,
        host=b,
        port=p,
        log_level="debug" if config.gateway.verbose else "info",
        access_log=config.gateway.verbose,
    )
