from __future__ import annotations

from pathlib import Path
from typing import Literal
from loguru import logger
from pydantic import Field, field_validator, BaseModel
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict, CliSubCommand, CliApp



class GlobalSettings(BaseSettings):
    """Configuration settings for the application.

    Settings model allows a .bingraphenv config file.
    Also, CLI arguments are implicitly parsed, as well as env variables.
    """
    model_config = SettingsConfigDict(
        env_prefix="BINGRAPH_",
        env_file=".bingraphenv",
        env_nested_delimiter="__", # for parsing from env (e.g., BINGRAPH_SERVER__HOST)
        cli_parse_args=True,       # automatic parsing of CLI options
        cli_implicit_flags=True,   # allows using "--no-x" bool flag modes
        cli_kebab_case=True,       # CLI options should be shown as kebab-case
    )

    root: Path = Field(..., description="The root directory for binaries")
    cfg_mode: Literal["none", "stateless", "stateful", "custom"] = Field(
        "stateless",
        description="CFG reconstruction mode and fallback strategy",
    )
    comments: CliImplicitFlag[bool] = Field(True, description="Appends comments to instructions when available")

    log_level: str = Field("INFO", description="Flag to define level for logging messages")  # type: ignore
    debug: CliImplicitFlag[bool] = Field(False, description="Flag to enable debug mode")

    @field_validator("root", mode="before")
    def validate_root(cls, v):
        """Makes sure provided root directory exists."""

        if not Path(v).exists():
            raise ValueError(f"root directory {v} does not exist")
        return Path(v)

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Makes sure log level is valid value."""

        upper_v = v.upper()
        valid_levels = logger._core.levels.keys()
        if upper_v not in valid_levels:
            raise ValueError(f"Must be one of: {', '.join(valid_levels)}")
        return upper_v


class ServerSettings(BaseModel):
    """Server mode settings."""

    host: str = Field("127.0.0.1", description="The host IP address for the server")
    port: int = Field(8000, description="The port number to run the server")

    def cli_cmd(self) -> None:
        pass

class ClientSettings(BaseModel):
    """Server mode settings."""

    endpoint: str = Field(..., description="Target API endpoint")
    payload: str | None = Field(None, description="JSON Payload data")

    filepath: str = Field(..., description="Path to target binary (relative to root directory for binaries)")
    function: str | None = Field(None, description="Function address (in int or 0x hex format)")
    format: str = Field("dot", description="Output format for graphs")

    def cli_cmd(self) -> None:
        pass


_settings: Settings | None = None


class Settings(GlobalSettings):
    """Root settings model."""

    server: CliSubCommand[ServerSettings]
    client: CliSubCommand[ClientSettings]

    def cli_cmd(self) -> None:
        """
        Pydantic executes this method immediately after successfully parsing the CLI.
        We intercept execution here to save the instance to our global singleton.
        """
        global _settings
        _settings = self  # Save this specific instantiated runtime object

        # Hand off execution to the typed subcommand
        CliApp.run_subcommand(self)


def get_settings() -> Settings:
    """Returns a cached, globally shared Settings instance,
    but defers CLI parsing until explicitly called.
    """
    global _settings
    if _settings is None:
        _settings = Settings(_cli_parse_args=True)
    return _settings
