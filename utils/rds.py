import psycopg2
from db.settings import get_settings


def get_connection():
    settings = get_settings()
    return psycopg2.connect(
        host=settings.ZEPHYRWERK_RDS_HOST,
        port=settings.ZEPHYRWERK_RDS_PORT,
        user=settings.ZEPHYRWERK_RDS_USER,
        password=settings.ZEPHYRWERK_RDS_PASSWORD,
        dbname=settings.ZEPHYRWERK_RDS_DB,
    )