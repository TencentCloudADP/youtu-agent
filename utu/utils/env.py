import importlib.metadata
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(raise_error_if_not_found=True), verbose=True, override=True)


class EnvUtils:
    @staticmethod
    def get_env(key: str, default: str | None = None) -> str | None:
        """Get the value of an environment variable. Raise an error if default is not privided (None)"""
        if default is None:
            res = os.getenv(key)
            if res is None:
                raise ValueError(f"Environment variable {key} is not set")
            return res
        return os.getenv(key, default)

    @staticmethod
    def assert_env(key: str | list[str]) -> None:
        if isinstance(key, list):
            for k in key:
                EnvUtils.assert_env(k)
        else:
            if not os.getenv(key):
                raise ValueError(f"Environment variable {key} is not set")

    @staticmethod
    def ensure_package(package_name: str) -> None:
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f"Package {package_name} is required but not installed!") from None
