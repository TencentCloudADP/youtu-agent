import base64
import pathlib

import requests


def encode_image(image_path: str) -> str:
    """Encode an image from a local path or URL to a base64 string."""
    if pathlib.Path(image_path).is_file():
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        response = requests.get(image_path)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return f"data:image/jpeg;base64,{image_base64}"
