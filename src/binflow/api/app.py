from __future__ import annotations

from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from binflow.api.settings import Settings
from binflow.core import load_project, list_function_symbols, render_cfg
from binflow.core.utils import resolve_under_root


def create_app(settings: Settings) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This function initializes the FastAPI application.
    It configures Jinja2 templates using the "templates" directory relative to the current file,
    and registers two endpoints:
      - /symtab: Returns an HTML page with the symbol table for a given binary file.
      - /cfg: Returns the control flow graph (CFG) as an SVG image for a specified function address.

    Args:
        settings (Settings): Application settings containing configuration such as the root directory.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(title="binflow")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    # logging setup
    logging.basicConfig(format=settings.logging_format, level=settings.logging_level, force=True)
    logging.info("Server initialized")

    def _resolve_path(filepath: str) -> Path:
        """Check input path for binary."""

        try:
            return resolve_under_root(settings.root, filepath)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/symtab", response_class=HTMLResponse)
    def symtab(request: Request, filepath: str = Query(...)) -> HTMLResponse:
        """
        Endpoint to display the symbol table of a binary file.

        This endpoint resolves the provided filepath under the application's root directory.
        It then loads the binary file as an angr project, retrieves the list of function symbols,
        and renders an HTML template ("symtab.html") populated with the symbols.

        Args:
            request (Request): The HTTP request object.
            filepath (str): Relative path to the binary file provided via query parameter.

        Returns:
            HTMLResponse: Rendered HTML page displaying the symbol table.
        """
        logging.info("Got request for /symtab")

        resolved = _resolve_path(filepath)
        project = load_project(resolved)
        symbols = list_function_symbols(project)

        return templates.TemplateResponse(
            request,
            "symtab.html",
            {
                "symbols": symbols,
                "filepath": filepath,
            },
        )

    def _resolve_faddr(faddr: str) -> int:
        """Make sure function addr is valid."""

        try:
            return int(faddr, 16) if faddr.startswith("0x") else int(faddr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid function address") from exc

    def _render_cfg(filepath: str, function: str, cfg_mode: bool) -> Response:
        """
        Return the control flow graph (CFG) of a specified function as an SVG image.

        This endpoint resolves the provided filepath and converts the function address from a string format.
        It then loads the binary file as an angr project and generates an SVG representation of the CFG for the specified function.
        If the function is not found or any error occurs, it returns an appropriate HTTP error.

        Args:
            filepath (str): Relative path to the binary file provided via query parameter.
            function (str): Function address as a string (hexadecimal with "0x" prefix or decimal).

        Returns:
            Response: An HTTP response containing the SVG image of the CFG with content type "image/svg+xml".
        """
        func_addr = _resolve_faddr(function)
        resolved = _resolve_path(filepath)
        project = load_project(resolved)

        try:
            svg = render_cfg(project, func_addr, cfg_mode=cfg_mode)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/cfg")
    def cfg(filepath: str = Query(...), function: str = Query(...)) -> Response:
        """Endpoint to return the fast CFG of a specified function as an SVG image."""

        return _render_cfg(filepath, function, cfg_mode=settings.cfg_mode)

    @app.get("/cfgfast")
    def cfgfast(filepath: str = Query(...), function: str = Query(...)) -> Response:
        """Endpoint to return the fast CFG of a specified function as an SVG image."""

        return _render_cfg(filepath, function, cfg_mode="fast")

    @app.get("/cfgemu")
    def cfgemu(filepath: str = Query(...), function: str = Query(...)) -> Response:
        """Endpoint to return the emulated CFG of a specified function as an SVG image."""

        return _render_cfg(filepath, function, cfg_mode=False)

    return app