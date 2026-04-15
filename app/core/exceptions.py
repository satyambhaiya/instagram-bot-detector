"""
Custom exceptions for BotScan.
Each exception maps to an HTTP status code via the handlers registered in main.py.
"""

from __future__ import annotations


class BotScanException(Exception):
    """Base exception for all BotScan errors."""
    status_code: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class InstagramUserNotFoundError(BotScanException):
    """The requested Instagram username does not exist."""
    status_code = 404
    detail = "Instagram user not found."


class InstagramPrivateAccountError(BotScanException):
    """Account is private — limited analysis possible."""
    status_code = 200   # not an error, but triggers reduced analysis
    detail = "Account is private. Analysis is based on public profile data only."


class InstagramProviderError(BotScanException):
    """Upstream Instagram data provider returned an error."""
    status_code = 503
    detail = "Instagram data provider is unavailable. Please try again later."


class RateLimitError(BotScanException):
    """Too many requests to the Instagram data provider."""
    status_code = 429
    detail = "Rate limit reached. Please wait before making another request."


class ModelNotReadyError(BotScanException):
    """ML model artifacts are missing or failed to load."""
    status_code = 503
    detail = "ML model is not ready. Run `python model/train.py` first."


class InvalidUsernameError(BotScanException):
    """Username contains invalid characters."""
    status_code = 422
    detail = "Invalid Instagram username."
