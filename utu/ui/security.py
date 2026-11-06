"""Access control shared by the WebUI servers.

- Origin: WebSocket and upload requests must come from the page's own origin, or from an origin listed in
  ``UTU_WEBUI_ALLOWED_ORIGINS`` (comma separated, ``*`` allows any).
- Token: when ``UTU_WEBUI_TOKEN`` is set, every request must carry it. Open ``/?token=<token>`` once; the server then
  stores it in an HttpOnly, SameSite=Strict cookie. The ``X-UTU-WebUI-Token`` header is accepted as well.
"""

import hmac
import ipaddress
import logging
from urllib.parse import urlparse

import tornado.web

from utu.utils import EnvUtils

TOKEN_COOKIE = "utu_webui_token"
TOKEN_HEADER = "X-UTU-WebUI-Token"


def get_token() -> str:
    return EnvUtils.get_env("UTU_WEBUI_TOKEN", "")


def get_allowed_origins() -> list[str]:
    raw = EnvUtils.get_env("UTU_WEBUI_ALLOWED_ORIGINS", "")
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def is_origin_allowed(handler: tornado.web.RequestHandler, origin: str | None) -> bool:
    """Same-origin requests and requests without an Origin header (non-browser clients) pass."""
    if not origin:
        return True
    allowed = get_allowed_origins()
    if "*" in allowed or origin.rstrip("/") in allowed:
        return True
    return urlparse(origin).netloc.lower() == handler.request.host.lower()


def _supplied_token(handler: tornado.web.RequestHandler) -> str:
    return (
        handler.get_query_argument("token", None)
        or handler.request.headers.get(TOKEN_HEADER)
        or handler.get_cookie(TOKEN_COOKIE)
        or ""
    )


def is_token_valid(handler: tornado.web.RequestHandler) -> bool:
    token = get_token()
    if not token:
        return True
    return hmac.compare_digest(_supplied_token(handler).encode(), token.encode())


def require_token(handler: tornado.web.RequestHandler) -> None:
    if not is_token_valid(handler):
        raise tornado.web.HTTPError(401, reason="Unauthorized")


class ProtectedStaticFileHandler(tornado.web.StaticFileHandler):
    """Serves the frontend; with a token configured, also turns ``?token=`` into the session cookie."""

    def prepare(self):
        require_token(self)
        token = get_token()
        if token and self.get_query_argument("token", None):
            self.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="Strict")


def ensure_safe_bind(ip: str) -> None:
    """Refuse to listen on a non-loopback address without a token, unless UTU_WEBUI_ALLOW_INSECURE=true."""
    try:
        loopback = ipaddress.ip_address(ip).is_loopback
    except ValueError:
        loopback = ip == "localhost"
    if loopback or get_token():
        return
    message = (
        f"WebUI listens on {ip} without UTU_WEBUI_TOKEN: anyone who can reach it can run agents, including "
        "agents with shell tools."
    )
    if EnvUtils.get_env("UTU_WEBUI_ALLOW_INSECURE", "false").lower() == "true":
        logging.warning(message + " Allowed by UTU_WEBUI_ALLOW_INSECURE=true.")
        return
    raise RuntimeError(
        message + " Set UTU_WEBUI_TOKEN, or set UTU_WEBUI_ALLOW_INSECURE=true if an authenticating reverse proxy "
        "already protects it."
    )
