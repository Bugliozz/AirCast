"""HTTP client with automatic rate-limit detection, exponential back-off,
jitter, and a circuit-breaker that saves the checkpoint before dying.

Usage
-----
    client = RateLimitedClient()
    resp = client.get("https://example.com/api", params={...})
    # If rate-limited the client sleeps transparently.
    # After MAX_CONSECUTIVE_FAILURES it trips the circuit breaker.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

from config import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF_S,
    JITTER_FACTOR,
    MAX_BACKOFF_S,
    MAX_CONSECUTIVE_FAILURES,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)

log = logging.getLogger(__name__)


# -- Exceptions ---------------------------------------------------------------

class CircuitBreakerOpen(Exception):
    """Too many consecutive failures — caller should save state and exit."""


class MaxRetriesExceeded(Exception):
    """All retry attempts for a single request exhausted."""


# -- Client -------------------------------------------------------------------

class RateLimitedClient:
    """Thin wrapper around ``requests.Session`` with retry + circuit-breaker."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._consecutive_failures: int = 0
        self._circuit_break_count: int = 0
        self._total_requests: int = 0
        self._total_retries: int = 0

    # -- public ---------------------------------------------------------------

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue a GET request with automatic retry on rate-limit / errors."""
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        backoff = INITIAL_BACKOFF_S

        for attempt in range(1, MAX_RETRIES + 1):
            self._total_requests += 1
            try:
                resp = self._session.get(url, **kwargs)

                if resp.status_code == 429:
                    wait = self._parse_retry_after(resp, backoff)
                    log.warning(
                        "429 on %s — sleeping %.0fs  [attempt %d/%d]",
                        url, wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    backoff = self._next_backoff(backoff)
                    self._total_retries += 1
                    self._record_failure()
                    continue

                if resp.status_code >= 500:
                    wait = self._jittered(backoff)
                    log.warning(
                        "HTTP %d on %s — sleeping %.0fs  [attempt %d/%d]",
                        resp.status_code, url, wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    backoff = self._next_backoff(backoff)
                    self._total_retries += 1
                    self._record_failure()
                    continue

                resp.raise_for_status()
                self._consecutive_failures = 0
                return resp

            except requests.exceptions.ConnectionError as exc:
                self._handle_transport_error("Connection error", url, exc, attempt, backoff)
                backoff = self._next_backoff(backoff)

            except requests.exceptions.Timeout as exc:
                self._handle_transport_error("Timeout", url, exc, attempt, backoff)
                backoff = self._next_backoff(backoff)

        raise MaxRetriesExceeded(f"Exhausted {MAX_RETRIES} attempts for {url}")

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_requests": self._total_requests,
            "total_retries": self._total_retries,
            "circuit_breaks": self._circuit_break_count,
        }

    # -- internal helpers -----------------------------------------------------

    def _handle_transport_error(
        self, label: str, url: str, exc: Exception, attempt: int, backoff: float,
    ) -> None:
        wait = self._jittered(backoff)
        log.warning(
            "%s on %s: %s — sleeping %.0fs  [attempt %d/%d]",
            label, url, exc, wait, attempt, MAX_RETRIES,
        )
        time.sleep(wait)
        self._total_retries += 1
        self._record_failure()

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._trip_circuit_breaker()

    def _trip_circuit_breaker(self) -> None:
        self._circuit_break_count += 1
        log.warning(
            "Circuit breaker #%d — exiting; next hourly run resumes from checkpoint.",
            self._circuit_break_count,
        )
        raise CircuitBreakerOpen(
            f"Circuit breaker tripped ({self._consecutive_failures} consecutive "
            f"failures) — save checkpoint and exit."
        )

    @staticmethod
    def _parse_retry_after(resp: requests.Response, fallback: float) -> float:
        raw = resp.headers.get("Retry-After")
        if raw is not None:
            try:
                return float(raw) + random.uniform(1, 5)
            except ValueError:
                pass
        return RateLimitedClient._jittered(fallback)

    @staticmethod
    def _jittered(base: float) -> float:
        return base * (1.0 + random.uniform(-JITTER_FACTOR, JITTER_FACTOR))

    @staticmethod
    def _next_backoff(current: float) -> float:
        return min(current * BACKOFF_MULTIPLIER, MAX_BACKOFF_S)
