"""Nobitex API-key authentication (Ed25519 request signing).

Nobitex has two credential types and they are *not* interchangeable:

1. **Login token** -- obtained from ``POST /auth/login/``, sent as
   ``Authorization: Token <hex>``. This is the older scheme; the docs show it
   as ``yourTOKENhereHEX...``.
2. **API key** -- created via ``POST /apikeys/create``, with scoped
   permissions (READ / TRADE / WITHDRAW), an optional IP allowlist and an
   expiry. It is **not** a bearer token: every request must carry an Ed25519
   signature.

An API key is a *pair*. The public key goes in the ``Nobitex-Key`` header and
the private key signs the request. Both are 44-character base64 strings, so
they look identical and are easy to mix up -- passing a public key as a
bearer token (the natural guess) returns 401 with no hint as to why.

Signature, per the docs::

    signature = base64(Ed25519(timestamp + method + url + body))

where ``url`` is the path *including* the query string, and ``body`` is the
raw request body (empty for GET).

Required headers::

    Nobitex-Key         public key
    Nobitex-Signature   the signature above
    Nobitex-Timestamp   current Unix time in seconds, UTC

Nobitex also asks bots to send ``User-Agent: TraderBot/<name>`` so their
support can identify automated traffic.
"""

from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Generator

import httpx
from cryptography.exceptions import InvalidKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class NobitexAuthError(RuntimeError):
    pass


def decode_key(value: str) -> bytes:
    """Decode a Nobitex key, accepting both base64 alphabets.

    Nobitex's own example private key contains ``-`` and ``_`` (base64url)
    while public keys are usually standard base64. Try both rather than
    guessing, and pad if the trailing ``=`` was stripped somewhere.
    """
    cleaned = value.strip().strip('"').strip("'")
    padded = cleaned + "=" * (-len(cleaned) % 4)

    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = decoder(padded)
        except (binascii.Error, ValueError):
            continue
        if len(raw) == 32:
            return raw

    raise NobitexAuthError(
        f"key is not 32 bytes of base64 (got {len(cleaned)} chars); "
        "expected a 44-character value like 'S5y19KewZzheCWCO4xqMcwwvtR8vQ-hHjE_cdjz-XxE='"
    )


def signing_payload(timestamp: int | str, method: str, url_path: str, body: str = "") -> bytes:
    """Build the exact byte string Nobitex signs.

    `url_path` must include the query string and must NOT include the scheme
    or host -- signing the full URL produces a valid-looking signature that
    the server rejects.
    """
    return f"{timestamp}{method.upper()}{url_path}{body}".encode()


class NobitexAPIKeyAuth(httpx.Auth):
    """httpx auth flow that signs every request with an Ed25519 API key."""

    requires_request_body = True

    def __init__(self, public_key: str, private_key: str):
        self.public_key = public_key.strip()
        try:
            self._signer = Ed25519PrivateKey.from_private_bytes(decode_key(private_key))
        except (InvalidKey, ValueError) as exc:
            raise NobitexAuthError(f"private key is not a valid Ed25519 seed: {exc}") from exc

        # A public key passed where the private one belongs still decodes to 32
        # bytes and still signs -- it just produces signatures the server
        # rejects as 401. Catch the obvious case early.
        if self.public_key == private_key.strip():
            raise NobitexAuthError(
                "public and private key are identical; an API key is a pair, and the "
                "private key is shown only once when the key is created"
            )

    def sign(self, method: str, url_path: str, body: bytes = b"") -> tuple[str, str]:
        timestamp = str(int(time.time()))
        payload = signing_payload(timestamp, method, url_path, body.decode() if body else "")
        signature = base64.b64encode(self._signer.sign(payload)).decode()
        return signature, timestamp

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        # Path + query, exactly as the docs specify.
        url_path = request.url.raw_path.decode()
        signature, timestamp = self.sign(request.method, url_path, request.content or b"")

        request.headers["Nobitex-Key"] = self.public_key
        request.headers["Nobitex-Signature"] = signature
        request.headers["Nobitex-Timestamp"] = timestamp
        # Nobitex asks bots to identify themselves in this format.
        request.headers.setdefault("User-Agent", "TraderBot/nbtrend")
        yield request
