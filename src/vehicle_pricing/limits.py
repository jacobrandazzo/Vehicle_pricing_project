"""Usage limits that keep the public website from running up the Claude bill.

Plain Python with no Streamlit, so the counting can be tested on its own. The
website keeps one DailyCounter for the whole server and a per-visit count in
its session state.
"""

import threading
from collections.abc import Callable
from datetime import UTC, date, datetime

MAX_CALLS_PER_SESSION = 20  # Claude calls one visit may make
MAX_CALLS_PER_DAY = 500  # Claude calls all visitors together may make per UTC day
MAX_DESCRIPTION_CHARS = 200  # longest description sent to Claude


def utc_today() -> date:
    """Today's date in UTC, so the day rolls over at the same moment wherever the server is."""
    return datetime.now(UTC).date()


class DailyCounter:
    """Counts Claude calls per UTC day, shared by every visitor's session.

    Streamlit runs each session in its own thread, so two visitors can call
    try_use() at the same moment. Checking the count and adding one are two
    steps; without the lock both could see 499, both pass, and 501 calls get
    through. The lock lets one thread at a time do both steps.

    `today` returns the current date. Tests pass a fake one to move the date on.
    """

    def __init__(self, limit: int = MAX_CALLS_PER_DAY, today: Callable[[], date] = utc_today):
        self._limit = limit
        self._today = today
        self._lock = threading.Lock()
        self._date = today()
        self._count = 0

    def try_use(self) -> bool:
        """Count one call and return True, or return False if today's limit is used up.

        The count starts again from 0 when the UTC date changes.
        """
        with self._lock:
            today = self._today()
            if today != self._date:
                self._date, self._count = today, 0
            if self._count >= self._limit:
                return False
            self._count += 1
            return True
