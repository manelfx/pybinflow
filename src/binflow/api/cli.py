from __future__ import annotations

import uvicorn

from .app import create_app
from .settings import Settings


def get_server_app():
    """
    Creates server app configured from CLI arguments.
    """
    settings = Settings()
    return create_app(settings)


def main() -> None:
    """
    Main entry point for running the binflow API server from the command-line.

    This function parses command-line arguments, configures the application settings based on the provided root
    directory or environment variables, creates the FastAPI application, and launches the server using uvicorn.
    Depending on 'reload' CLI argument, it allows service reloading on sources changes (dev mode).
    """
    settings = Settings()    # CLI arguments are implicitly parsed

    uvicorn.run(
        "binflow.api.cli:get_server_app" if settings.reload else create_app(settings),
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        factory=settings.reload
    )


if __name__ == "__main__":
    main()
