"""Typed configuration loaded from config/*.yaml plus environment/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader -- avoids a dependency and never clobbers real env."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    nobitex: str
    src: str
    dst: str
    tradingview: str | None = None
    amount_step: float = 1e-8
    price_step: float = 1.0
    enabled: bool = True
    multiplier: int = 1
    """Base-asset units per quoted unit.

    Nobitex quotes `1K_SHIBIRT` per 1,000 SHIB and `1M_PEPEIRT` per 1,000,000
    PEPE, while the global feed is always per 1 unit. The fair rial price is
    therefore `global_usd * fx * multiplier`. Leaving this at 1 on a scaled
    market makes the basis check see a 1000x dislocation and refuse to trade
    it, every cycle, silently.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymbolSpec:
        return cls(
            nobitex=d["nobitex"],
            src=d["src"],
            dst=d["dst"],
            tradingview=d.get("tradingview"),
            amount_step=float(d.get("amount_step", 1e-8)),
            price_step=float(d.get("price_step", 1.0)),
            enabled=bool(d.get("enabled", True)),
            multiplier=int(d.get("multiplier", 1)),
        )


@dataclass(frozen=True, slots=True)
class Credentials:
    api_token: str | None
    ws_auth_param: str | None
    testnet: bool
    api_key: str | None = None
    api_secret: str | None = None

    @property
    def uses_api_key(self) -> bool:
        """API keys are Ed25519-signed, not bearer tokens."""
        return bool(self.api_key and self.api_secret)

    @property
    def can_trade(self) -> bool:
        return bool(self.api_token) or self.uses_api_key

    def require_token(self) -> str:
        if self.uses_api_key:
            return self.api_key  # signed per request, not a bearer token
        if not self.api_token:
            raise RuntimeError(
                "No Nobitex credentials. Either set NOBITEX_API_TOKEN (a login token from "
                "POST /auth/login/), or set NOBITEX_API_KEY and NOBITEX_API_SECRET (an API "
                "key pair from POST /apikeys/create, signed with Ed25519). An API key's "
                "public half used as NOBITEX_API_TOKEN returns 401 -- they are different "
                "schemes. Public market data works without either."
            )
        return self.api_token


@dataclass(slots=True)
class Config:
    raw: dict[str, Any]
    universe: list[SymbolSpec]
    fx: SymbolSpec
    creds: Credentials
    mode: str

    # -- nested section accessors ------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def strategy(self) -> dict[str, Any]:
        return self.raw["strategy"]

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw["risk"]

    @property
    def costs(self) -> dict[str, Any]:
        return self.raw["costs"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw["execution"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    # -- endpoints, testnet-aware ------------------------------------------
    @property
    def rest_url(self) -> str:
        nb = self.data["nobitex"]
        return nb["testnet_rest_url"] if self.creds.testnet else nb["rest_url"]

    @property
    def ws_url(self) -> str:
        nb = self.data["nobitex"]
        return nb["testnet_ws_url"] if self.creds.testnet else nb["ws_url"]

    def symbol(self, nobitex_symbol: str) -> SymbolSpec:
        for spec in [*self.universe, self.fx]:
            if spec.nobitex == nobitex_symbol:
                return spec
        raise KeyError(f"{nobitex_symbol} is not in config/universe.yaml")

    @property
    def enabled_symbols(self) -> list[SymbolSpec]:
        return [s for s in self.universe if s.enabled]


def load_config(
    config_path: Path | str | None = None,
    universe_path: Path | str | None = None,
) -> Config:
    _load_dotenv(PROJECT_ROOT / ".env")

    config_path = Path(config_path or PROJECT_ROOT / "config" / "config.yaml")
    universe_path = Path(universe_path or PROJECT_ROOT / "config" / "universe.yaml")

    raw = yaml.safe_load(config_path.read_text())
    uni = yaml.safe_load(universe_path.read_text())

    creds = Credentials(
        api_token=os.environ.get("NOBITEX_API_TOKEN") or None,
        ws_auth_param=os.environ.get("NOBITEX_WS_AUTH_PARAM") or None,
        testnet=os.environ.get("NOBITEX_TESTNET", "0") == "1",
        api_key=os.environ.get("NOBITEX_API_KEY") or None,
        api_secret=os.environ.get("NOBITEX_API_SECRET") or None,
    )

    return Config(
        raw=raw,
        universe=[SymbolSpec.from_dict(d) for d in uni["symbols"]],
        fx=SymbolSpec.from_dict(uni["fx"]),
        creds=creds,
        mode=os.environ.get("NBTREND_MODE", "paper").lower(),
    )
