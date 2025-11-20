import os
from dataclasses import dataclass


@dataclass
class EnvConfig:
    MONGODB_CONNECTION_STRING: str

    @classmethod
    def from_env(cls) -> "EnvConfig":
        conn = os.getenv("MONGODB_CONNECTION_STRING")

        if conn is None:
            raise EnvironmentError("MONGODB_CONNECTION_STRING not set in environment")

        return cls(MONGODB_CONNECTION_STRING=conn)
