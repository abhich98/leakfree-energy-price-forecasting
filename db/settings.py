from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ZEPHYRWERK_RDS_HOST: str
    ZEPHYRWERK_RDS_PORT: int = 5432
    ZEPHYRWERK_RDS_DB: str
    ZEPHYRWERK_RDS_USER: str
    ZEPHYRWERK_RDS_PASSWORD: str

    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_ENDPOINT_URL: str | None = None

    ZEPHYRWERK_AWS_BUCKET_NAME: str

    ZEPHYRWERK_SMARD_BASE_URL: str

    ZEPHYRWERK_OPENMETEO_HISTORICAL_URL: str
    ZEPHYRWERK_OPENMETEO_FORECAST_URL: str
    ZEPHYRWERK_OPENMETEO_HISTORICAL_FORECAST_URL: str
    ZEPHYRWERK_OPENMETEO_SINGLE_RUNS_FORECAST_URL: str

    ZEPHYRWERK_API_HOST: str
    ZEPHYRWERK_API_PORT: int

    ZEPHYRWERK_DASHBOARD_API_URL: str


    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, env_file_encoding="utf-8")
        
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.ZEPHYRWERK_RDS_USER}:{self.ZEPHYRWERK_RDS_PASSWORD}"
            f"@{self.ZEPHYRWERK_RDS_HOST}:{self.ZEPHYRWERK_RDS_PORT}/{self.ZEPHYRWERK_RDS_DB}"
        )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
    