from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, CliImplicitFlag


class Settings(BaseSettings):
    """Configuration settings for the application.

    Settings model allows a .binflowenv config file.
    Also, CLI arguments are implicitly parsed, as well as env variables.
    """
    model_config = ConfigDict(
        env_file=".binflowenv",
        cli_parse_args=True,       # automatic parsing of CLI options
        alias_generator=lambda field_name: field_name.replace("_", "-")
                                   # swap underscores for dashes across all fields
    )

    root: Path = Field(
        ..., env="BINFLOW_ROOT", description="The root directory for binaries")
    host: str = Field(
        "127.0.0.1", env="BINFLOW_HOST", description="The host IP address for the server")
    port: int = Field(
        8000, env="BINFLOW_PORT", description="The port number to run the server")
    reload: CliImplicitFlag[bool] = Field(
        False, env="BINFLOW_RELOAD", description="Flag to enable auto-reloading of the server")

    cfg_mode: Literal["fast", "emulated"] = Field(
        "fast", env="BINFLOW_CFG_MODE", description="CFG reconstruction mode")

    logging_level: str = Field(
        "INFO", env="BINFLOW_LOGGING_LEVEL", description="Flag to define logging level")
    logging_format: str = Field(
        "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
        env="BINFLOW_LOGGING_FORMAT", description="Flag to define logging format")

    @field_validator("root", mode="before")
    def validate_root(cls, v):
        """
        Makes sure provided root directory exists.
        """
        if not Path(v).exists():
            raise ValueError(f"root directory {v} does not exist")
        return Path(v)
