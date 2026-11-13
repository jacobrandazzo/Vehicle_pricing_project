"""The daily counter allows exactly the daily limit, then refuses until the UTC date changes."""

from datetime import date

from vehicle_pricing.limits import MAX_CALLS_PER_DAY, DailyCounter


class FakeToday:
    """Stands in for utc_today(): returns `.value`, which a test can move on."""

    def __init__(self, value: date):
        self.value = value

    def __call__(self) -> date:
        return self.value


def test_allows_exactly_the_daily_limit_then_refuses():
    counter = DailyCounter(today=FakeToday(date(2026, 11, 12)))

    allowed = [counter.try_use() for _ in range(MAX_CALLS_PER_DAY)]

    assert all(allowed)
    assert counter.try_use() is False
    assert counter.try_use() is False  # stays refused for the rest of the day


def test_resets_on_a_new_utc_date():
    today = FakeToday(date(2026, 11, 12))
    counter = DailyCounter(limit=3, today=today)
    for _ in range(3):
        counter.try_use()
    assert counter.try_use() is False

    today.value = date(2026, 11, 13)

    assert [counter.try_use() for _ in range(4)] == [True, True, True, False]
