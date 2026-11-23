"""Server configuration, read from environment variables at startup.

``DOBERMANN_DEVICE``
    The device to run the model on: ``auto`` (the default) picks a GPU if
    one is available, ``cuda`` requires one and ``cpu`` never uses one.

Host and port are set on the uvicorn command line.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEVICE_VARIABLE = "DOBERMANN_DEVICE"

# Maps each allowed value to the device passed to the embedder.
DEVICES: dict[str, str | None] = {"auto": None, "cuda": "cuda", "cpu": "cpu"}


@dataclass(frozen=True)
class Settings:
    device: str | None = None
    """The device for the embedder, or ``None`` to pick one automatically."""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Settings":
        """Read the settings from the environment.

        Raises:
            ValueError: If a variable has an invalid value.
        """
        device = environ.get(DEVICE_VARIABLE, "auto").strip().lower()
        if device not in DEVICES:
            raise ValueError(
                f"Invalid {DEVICE_VARIABLE}={environ[DEVICE_VARIABLE]!r}: "
                f"expected one of {', '.join(DEVICES)}"
            )
        return cls(device=DEVICES[device])
