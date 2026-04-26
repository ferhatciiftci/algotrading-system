"""
data/schema.py
──────────────
Pydantic models that define the expected shape of raw and clean data.
Used for validation at ingestion time and after cleaning.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    # Provide minimal stubs so the module can be imported without pydantic.
    # Validation will not run — only affects the data-ingestion pipeline.
    _HAS_PYDANTIC = False
    class BaseModel:  # type: ignore
        def __init_subclass__(cls, **kw): pass
        def __init__(self, **data): self.__dict__.update(data)
        def model_dump(self): return self.__dict__.copy()
    def Field(*a, **kw): return None  # type: ignore
    def field_validator(*a, **kw): return lambda f: f  # type: ignore
    def model_validator(*a, **kw): return lambda f: f  # type: ignore


# ─── Raw bar (as delivered by a data vendor) ──────────────────────────────────

class RawBar(BaseModel):
    symbol    : str
    timestamp : datetime
    open      : float = Field(gt=0)
    high      : float = Field(gt=0)
    low       : float = Field(gt=0)
    close     : float = Field(gt=0)
    volume    : float = Field(ge=0)
    # Optional vendor fields
    adj_close : Optional[float] = None
    vwap      : Optional[float] = None

    @field_validator("symbol")
    @classmethod
    def symbol_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must not be empty")
        return v.upper().strip()

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def ohlc_consistency(self) -> "RawBar":
        if self.high < self.low:
            raise ValueError(f"high < low for {self.symbol} @ {self.timestamp}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open outside [low, high] for {self.symbol}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close outside [low, high] for {self.symbol}")
        return self


# ─── Clean bar (after adjustments and quality checks) ─────────────────────────

class CleanBar(BaseModel):
    symbol         : str
    timestamp      : datetime
    open           : float = Field(gt=0)
    high           : float = Field(gt=0)
    low            : float = Field(gt=0)
    close          : float = Field(gt=0)
    volume         : float = Field(ge=0)
    adj_close      : float = Field(gt=0)
    adj_factor     : float = Field(gt=0, description="Cumulative adjustment factor")
    source         : str   = "unknown"
    quality_flags  : int   = 0          # bitmask – see DataQualityFlag

    @model_validator(mode="after")
    def ohlc_consistency(self) -> "CleanBar":
        if self.high < self.low:
            raise ValueError(f"high < low for {self.symbol}")
        return self


# ─── Corporate action ─────────────────────────────────────────────────────────

class CorporateAction(BaseModel):
    symbol      : str
    ex_date     : datetime      # the date the action takes effect
    action_type : Literal["split", "dividend", "merger", "rename"]
    factor      : float         # for split: new/old; for dividend: $ per share
    new_symbol  : Optional[str] = None   # for rename/merger

    @field_validator("factor")
    @classmethod
    def factor_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Corporate action factor must be positive")
        return v


# ─── Symbol mapping ───────────────────────────────────────────────────────────

class SymbolMapping(BaseModel):
    """Maps a vendor symbol to a canonical internal symbol."""
    vendor_symbol    : str
    canonical_symbol : str
    valid_from       : datetime
    valid_to         : Optional[datetime] = None
    exchange         : str = "UNKNOWN"
