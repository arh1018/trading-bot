"""Nobitex API-key authentication.

Nobitex has two credential schemes that look nothing alike in behaviour but
identical on the page: a login token (bearer) and an API key pair (Ed25519
request signing). Passing an API key's public half as a bearer token is the
natural guess and returns a bare 401 with no explanation, so these tests pin
the distinction down.
"""

from __future__ import annotations

import base64
import time

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nbtrend.data.nobitex_auth import (
    NobitexAPIKeyAuth,
    NobitexAuthError,
    decode_key,
    signing_payload,
)

# Nobitex's own documented example keys.
DOC_PRIVATE = "S5y19KewZzheCWCO4xqMcwwvtR8vQ-hHjE_cdjz-XxE="   # base64url
DOC_PUBLIC = "5XOCQZSPLQM4MiLzuUnZoBuqgYgTKl40W2X5j1pxfIA="     # standard base64


def test_decodes_both_base64_alphabets():
    """The documented private key is base64url; public keys are standard."""
    assert len(decode_key(DOC_PRIVATE)) == 32
    assert len(decode_key(DOC_PUBLIC)) == 32


def test_decode_tolerates_missing_padding():
    assert len(decode_key(DOC_PUBLIC.rstrip("="))) == 32


def test_decode_rejects_a_non_key():
    with pytest.raises(NobitexAuthError, match="32 bytes"):
        decode_key("not-a-key")


def test_signing_payload_is_timestamp_method_url_body():
    assert signing_payload(1700000000, "GET", "/users/profile") == b"1700000000GET/users/profile"
    assert signing_payload(1700000000, "get", "/x") == b"1700000000GET/x"
    assert (
        signing_payload(1, "POST", "/market/orders/add", '{"a":1}')
        == b'1POST/market/orders/add{"a":1}'
    )


def test_signing_payload_keeps_the_query_string():
    """The docs sign `/market/orders/list?fromId=123`, query included."""
    payload = signing_payload(1, "GET", "/market/orders/list?fromId=123")
    assert b"?fromId=123" in payload


def test_signature_verifies_against_the_public_key():
    signer = Ed25519PrivateKey.generate()
    private_b64 = base64.b64encode(signer.private_bytes_raw()).decode()

    auth = NobitexAPIKeyAuth("some-public-key", private_b64)
    signature, timestamp = auth.sign("GET", "/users/profile")

    signer.public_key().verify(
        base64.b64decode(signature), signing_payload(timestamp, "GET", "/users/profile")
    )


def test_identical_key_pair_is_rejected():
    """A public key signs happily and produces 401s forever; catch it early."""
    with pytest.raises(NobitexAuthError, match="identical"):
        NobitexAPIKeyAuth(DOC_PRIVATE, DOC_PRIVATE)


def test_auth_flow_sets_all_three_headers():
    signer = Ed25519PrivateKey.generate()
    auth = NobitexAPIKeyAuth(DOC_PUBLIC, base64.b64encode(signer.private_bytes_raw()).decode())

    request = httpx.Request("GET", "https://apiv2.nobitex.ir/users/profile")
    signed = next(auth.auth_flow(request))

    assert signed.headers["Nobitex-Key"] == DOC_PUBLIC
    assert signed.headers["Nobitex-Signature"]
    assert abs(int(signed.headers["Nobitex-Timestamp"]) - time.time()) < 5
    # Nobitex asks bots to identify themselves in this format.
    assert signed.headers["User-Agent"].startswith("TraderBot/")


def test_auth_flow_signs_path_not_full_url():
    """Signing scheme+host produces a valid signature the server rejects."""
    signer = Ed25519PrivateKey.generate()
    auth = NobitexAPIKeyAuth(DOC_PUBLIC, base64.b64encode(signer.private_bytes_raw()).decode())

    request = httpx.Request("GET", "https://apiv2.nobitex.ir/market/orders/list?fromId=7")
    signed = next(auth.auth_flow(request))

    expected = signing_payload(
        signed.headers["Nobitex-Timestamp"], "GET", "/market/orders/list?fromId=7"
    )
    signer.public_key().verify(base64.b64decode(signed.headers["Nobitex-Signature"]), expected)


def test_credentials_distinguish_the_two_schemes(monkeypatch):
    from nbtrend.config import Credentials

    bearer = Credentials(api_token="deadbeef", ws_auth_param=None, testnet=False)
    assert bearer.can_trade and not bearer.uses_api_key

    keypair = Credentials(
        api_token=None, ws_auth_param=None, testnet=False,
        api_key=DOC_PUBLIC, api_secret=DOC_PRIVATE,
    )
    assert keypair.can_trade and keypair.uses_api_key

    # A public key alone is not enough -- signing needs the private half.
    half = Credentials(api_token=None, ws_auth_param=None, testnet=False, api_key=DOC_PUBLIC)
    assert not half.uses_api_key
    assert not half.can_trade


def test_require_token_explains_both_schemes():
    from nbtrend.config import Credentials

    empty = Credentials(api_token=None, ws_auth_param=None, testnet=False)
    with pytest.raises(RuntimeError, match="NOBITEX_API_KEY"):
        empty.require_token()
