"""Shared Binance USD-M perpetual universe for the conditional wick engine."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_ASSETS = ("ETHUSDT", "BTCUSDT", "SOLUSDT", "UNIUSDT", "NEARUSDT")
DEFAULT_LIBRARY_DIR_NAME = "conditional_path_library_5y_5assets"
DEFAULT_ONE_MINUTE_ASSET = "SOLUSDT"
# Retained as a compatibility name for scripts and existing SOL fixtures.
SOL_ONE_MINUTE_ASSET = DEFAULT_ONE_MINUTE_ASSET
SUPPORTED_DASHBOARD_TIMEFRAMES = ("1m", "5m", "15m")


def default_library_dir(root: Path) -> Path:
    """Return the versioned library location for the configured asset universe."""
    return root / "data" / DEFAULT_LIBRARY_DIR_NAME


def one_minute_library_dir(root: Path, asset: str) -> Path:
    """Return the isolated one-minute V1 trajectory library for one supported asset."""
    if asset not in SUPPORTED_ASSETS:
        raise ValueError(f"Unsupported one-minute asset: {asset}")
    return root / "data" / f"conditional_path_library_{asset.removesuffix('USDT').lower()}_1m_5y"


def sol_one_minute_library_dir(root: Path) -> Path:
    """Compatibility alias for the existing SOLUSDT one-minute library."""
    return one_minute_library_dir(root, DEFAULT_ONE_MINUTE_ASSET)


def assets_for_timeframe(timeframe: str) -> tuple[str, ...]:
    """Every configured asset has an independent 1m source and library."""
    if timeframe in SUPPORTED_DASHBOARD_TIMEFRAMES:
        return SUPPORTED_ASSETS
    raise ValueError(f"Unsupported dashboard timeframe: {timeframe}")


def asset_indicator_column(asset: str) -> str:
    """Return the stable V2 one-hot feature name for one supported asset."""
    return f"asset_is_{asset.removesuffix('USDT').lower()}"


ASSET_CATEGORICAL_FEATURE_COLUMNS = tuple(asset_indicator_column(asset) for asset in SUPPORTED_ASSETS)
