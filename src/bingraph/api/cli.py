from __future__ import annotations
import json

from fastapi.testclient import TestClient
from fastapi.routing import APIRoute
from pydantic_settings import CliApp
import uvicorn

from bingraph.helpers import get_settings, Settings
from .app import create_app


def main() -> None:
    """
    Main entry point for running the bingraph API server from the command-line.

    This function parses command-line arguments, configures the application settings based on the provided root
    directory or environment variables, creates the FastAPI application, and launches the server using uvicorn.
    Depending on 'debug' CLI argument, it allows service reloading on sources changes (dev mode).
    """
    CliApp.run(Settings)
    settings = get_settings()

    if settings.server:
        # If we are debugging the server, mute uvicorn's redundant error dumps
        # so they don't clobber our clean Loguru console outputs.
        log_config = uvicorn.config.LOGGING_CONFIG
        if settings.debug:
            log_config["loggers"]["uvicorn.error"]["level"] = "CRITICAL"

        uvicorn.run(
            "bingraph.api.app:create_app" if settings.debug else create_app(),
            host=settings.server.host,
            port=settings.server.port,
            reload=settings.debug,
            factory=settings.debug,
            log_config=log_config,
        )
    else:
        client_settings = settings.client
        if client_settings is None:
            raise RuntimeError(
                "Client settings are required when server mode is disabled"
            )

        app = create_app()
        client = TestClient(app, raise_server_exceptions=settings.debug)

        # build correct endpoint path
        api_endpoint = client_settings.endpoint
        if not api_endpoint.startswith("/"):
            api_endpoint = f"/{api_endpoint}"
        if not api_endpoint.startswith("/api"):
            api_endpoint = f"/api{api_endpoint}"
        route_paths = [
            route.path for route in app.routes if isinstance(route, APIRoute)
        ]
        assert api_endpoint in route_paths, f"Invalid endpoint {api_endpoint}"

        # build query parameters
        # all client settings attributes except endpoint and payload are considered parameters
        query_params = {
            key: value
            for key, value in vars(client_settings).items()
            if value is not None and key not in ["endpoint", "payload"]
        }

        # build payload info (if needed)
        body_data = client_settings.payload
        if body_data:
            try:
                body_data = json.loads(body_data)
            except json.JSONDecodeError:
                # Fall back to raw string if it's not JSON
                pass

        # TODO: as of today, no payloads allowed, only GET endpoints considered
        # json=body_data if isinstance(body_data, dict) else None,
        # data=body_data if isinstance(body_data, str) else None
        response = client.get(api_endpoint, params=query_params)
        print(json.dumps(response.json()))


if __name__ == "__main__":
    main()
