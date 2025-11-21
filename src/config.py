import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Final


@dataclass(frozen=True)
class EnvConfig:
    MONGODB_CONNECTION_STRING: str
    EMAIL_ADDRESS: str
    EMAIL_PASSWORD: str 

    @classmethod
    def from_env(cls) -> "EnvConfig":
        load_dotenv()
        
        conn = os.getenv("MONGODB_CONNECTION_STRING")
        email_address = os.getenv("EMAIL_ADDRESS")
        email_password = os.getenv("EMAIL_PASSWORD")

        if conn is None:
            raise EnvironmentError("MONGODB_CONNECTION_STRING not set in environment")

        if email_address is None:
            raise EnvironmentError("EMAIL_ADDRESS not set in environment")
        
        if email_password is None:
            raise EnvironmentError("EMAIL_PASSWORD not set in environment")

        return cls(MONGODB_CONNECTION_STRING=conn, 
                   EMAIL_ADDRESS=email_address, 
                   EMAIL_PASSWORD=email_password,)


ENV_CONFIG: Final[EnvConfig] = EnvConfig.from_env()