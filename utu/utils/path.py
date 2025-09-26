import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from jinja2 import Environment, FileSystemLoader, Template


def get_package_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


DIR_ROOT = get_package_path()
CACHE_DIR = DIR_ROOT / ".cache"


class FileUtils:
    @staticmethod
    def is_web_url(url: str) -> bool:
        parsed_url = urlparse(url)
        return all([parsed_url.scheme, parsed_url.netloc])

    @staticmethod
    def get_file_ext(file_path: str) -> str:
        if FileUtils.is_web_url(file_path):
            return pathlib.Path(urlparse(file_path).path).suffix
        return pathlib.Path(file_path).suffix

    @staticmethod
    def download_file(url: str, save_path: str = None) -> str:
        """Download file from web. Return the saved path"""
        # if not save_path, use tempfile
        if not save_path:
            save_path = tempfile.NamedTemporaryFile(
                suffix=FileUtils.get_file_ext(url),
                delete=False,
            ).name
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path

    @staticmethod
    def get_file_md5(file_path: str) -> str:
        """Clac md5 for local or web file"""
        hash_md5 = hashlib.md5()
        if FileUtils.is_web_url(file_path):
            with requests.get(file_path, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=4096):
                    hash_md5.update(chunk)
        else:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def load_yaml(file_path: pathlib.Path | str) -> dict[str, Any]:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with file_path.open() as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_prompts(fn: str | pathlib.Path) -> dict[str, str]:
        """Load prompts from yaml file.

        - Default path: `DIR_ROOT / "utu/prompts" / fn`
        """
        if isinstance(fn, str):
            if not fn.endswith(".yaml"):
                fn += ".yaml"
            fn = DIR_ROOT / "utu" / "prompts" / fn
        assert fn.exists(), f"File {fn} does not exist!"
        with fn.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_jinja_env(directory: str | pathlib.Path) -> Environment:
        if isinstance(directory, str):
            directory = DIR_ROOT / "utu" / "prompts" / directory
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        return Environment(loader=FileSystemLoader(directory))

    @staticmethod
    def get_jinja_template(template_path: str | pathlib.Path) -> Template:
        if isinstance(template_path, str):
            if not template_path.endswith(".j2"):
                template_path += ".j2"
            template_path = DIR_ROOT / "utu" / "prompts" / template_path
        if not template_path.exists():
            raise FileNotFoundError(f"File {template_path} does not exist")
        with template_path.open(encoding="utf-8") as f:
            return Template(f.read())

    @staticmethod
    def get_jinja_template_str(template_str: str) -> Template:
        return Template(template_str)

    @staticmethod
    def load_json(file_path: str | pathlib.Path) -> dict[str, Any]:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with file_path.open() as f:
            return json.load(f)

    @staticmethod
    def load_json_data(file_path: str | pathlib.Path) -> list[dict[str, Any]]:
        if isinstance(file_path, str) and not os.path.exists(file_path):
            file_path = DIR_ROOT / "utu" / "data" / file_path
        return FileUtils.load_json(file_path)

    @staticmethod
    def save_json(file_path: str | pathlib.Path, data: dict[str, Any]) -> None:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
