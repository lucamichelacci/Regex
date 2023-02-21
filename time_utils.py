import numpy as np


def is_leap(year: int) -> bool:
    """Asserts if a year is a leap year."""
    return np.logical_and(year % 4 == 0, np.logical_or(year % 100 != 0, year % 400 == 0))


def get_npdatetime64(year: int, month: int, day: int) -> np.datetime64:
    """ Returns a np.datetime64 object associated with the input year, date and day."""
    return np.datetime64(f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)}')


def get_weekday(date: np.datetime64) -> int:
    """ Returns the weekday of a np.datetime64 object."""
    return ((date.astype('datetime64[D]').view('int64') - 4) % 7).view('int64')


def get_day(date: np.datetime64) -> int:
    """ Returns the day of a np.datetime64 object."""
    return (date - date.astype('datetime64[M]') + 1).view('int64')


def get_month(date: np.datetime64) -> int:
    """ Returns the month of a np.datetime64 object."""
    return date.astype('datetime64[M]').astype(int) % 12 + 1


def get_year(date: np.datetime64) -> int:
    """ Returns the year of a np.datetime64 object."""
    return date.astype('datetime64[Y]').astype(int) + 1970


def is_eom(date: np.datetime64) -> bool:
    """ Asserts if a date is eom."""
    return get_month(date + np.timedelta64(1, '1D')) == get_month(date) + 1


# FIXME: Function is not working properly with arrays
def get_feb_eom(year: int) -> np.datetime64:
    """ Returns the last day of february for a given year."""
    return get_npdatetime64(year, 2, 28) + np.timedelta64(is_leap(year) * 1, 'D')


def get_easter(year: int):
    """ Returns the month and day of easter of the given year."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1
    return month, day


def get_good_friday(year: int):
    """ Returns the month and day of good friday of the given year."""
    easter_month, easter_day = get_easter(year)
    good_friday = get_npdatetime64(year, easter_month, easter_day) - np.timedelta64(2, 'D')
    return get_month(good_friday), get_day(good_friday)


# TODO: Vectorize this part and add holidays to handle
def build_month(start_week_day: int, num_days: int) -> np.matrix:
    """Returns a calendar month as a matrix."""
    days = np.arange(1, num_days + 1)
    month = np.pad(days, (start_week_day, 42 - len(days) - start_week_day), 'constant')
    return np.matrix(month.reshape((6, 7)))

# TODO: Fix this function
def build_calendar(year: int) -> np.array:
    """Method to build a calendar year. The calendar year is handled as a list of matrices."""
    days_per_month = np.array([31, 28 + 1 * is_leap(year), 31, 30, 31, 30, 31, 31, 30, 31, 31, 31])
    leap_adj = is_leap(year) * np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    bom1900 = np.array([0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5])  # Weekday of each month of the year 1900
    weekdays_offset = ((year - 1900) + np.sum(np.vectorize(is_leap)(np.arange(1900, year)))) % 7
    bomyear = ((weekdays_offset + leap_adj) + bom1900) % 7
    return bomyear, days_per_month


# holiday rules for NYSE (nth, week_day, month)
holiday_rules = np.array([(3, 0, 1), (3, 0, 2), (-1, 0, 5), (1, 0, 9), (4, 3, 11)])

# holiday dates for NYSE (month, day)
holiday_dates = np.array([(1, 1), (6, 19), (7, 4), (12, 25), get_good_friday], dtype='object')


def holiday_from_rule(holidays: list, nth: int, week_day: int, month: int, year: int) -> np.datetime64:
    """Returns the date of a holiday based on a rule - e.g. the nth monday of a month. The
     method also accepts negative nth parameters."""
    calendar = build_calendar(year)
    month_array = np.array(calendar[month - 1][:, week_day])
    day = month_array[month_array != 0][nth - 1 + 1 * (abs(nth) != nth)]
    holiday = get_npdatetime64(year, month, day)
    return holiday


def holiday_from_date(year: int, month: int, day: int, weekend_roll: bool = True) -> np.datetime64:
    """Returns the date of a holiday based on a specified date."""
    holiday = get_npdatetime64(year, month, day)
    weekday = get_weekday(holiday)
    if weekday < 5:
        return holiday
    elif weekday == 5 and weekend_roll:
        return holiday - np.timedelta64(1, 'D')
    elif weekday == 6 and weekend_roll:
        return holiday + np.timedelta64(1, 'D')


# TODO: Finish this function        
def get_holidays(year: int, holiday_rules, holiday_dates, weekend_roll: bool = True) -> np.array:
    pass


def build_schedule(effective: np.datetime64, maturity: np.datetime64, freq: int, roll: str = 'forward',
                   holidays: list = [], gen_backward: bool = False, eom: bool = True, first: np.datetime64 = None,
                   next_to_last: np.datetime64 = None, bump_maturity: bool = False) -> np.array:
    """Returns a payment schedule for fixed income instruments."""

    if gen_backward and next_to_last is None:
        next_to_last = maturity.astype('datetime64[M]') - np.timedelta64(freq, 'M')
        next_to_last = next_to_last.astype('datetime64[D]') + np.timedelta64(get_day(maturity) - 1, 'D')
        next_to_last = np.busday_offset(next_to_last, 0, roll=roll, holidays=holidays)

    elif gen_backward is False and first is None:
        first = effective.astype('datetime64[M]') + np.timedelta64(freq, 'M')
        first = first.astype('datetime64[D]') + np.timedelta64(get_day(effective) - 1, 'D')
        first = np.busday_offset(first, 0, roll, holidays=holidays)

    if gen_backward:
        start, end = next_to_last, effective
        flag = -1

    else:
        start, end = first, maturity
        flag = 1

    eom_adj = (is_eom(start) * eom)
    schedule = np.arange(start.astype('datetime64[M]'), end.astype('datetime64[M]'), np.timedelta64(flag * freq, 'M'))
    schedule = (schedule + np.timedelta64(1 * eom_adj, 'M')).astype('datetime64[D]')
    schedule += np.timedelta64((1 - eom_adj) * get_day(effective) - eom_adj - 1, 'D')
    schedule = np.busday_offset(schedule, 0, roll=roll, holidays=holidays)
    if bump_maturity:
        schedule = np.append(schedule, [effective, np.busday_offset(maturity, 0, roll=roll, holidays=holidays)])
    else:
        schedule = np.append(schedule, [effective, maturity])
    return np.sort(schedule)


def act_360(start_date: np.datetime64, end_date: np.datetime64) -> float:
    """ACT 360 day count standard."""
    actual_days = (end_date - start_date).view('int64')
    return actual_days / 360.0


def act_365_fixed(start_date: np.datetime64, end_date: np.datetime64) -> float:
    """ ACT 365 FIXED day count standard."""
    actual_days = (end_date - start_date).view('int64')
    return actual_days / 365.0


def thirty_360_isda(start_date: np.datetime64, end_date: np.datetime64) -> float:
    """ THIRTY 360 ISDA day count standard."""
    start_year = get_year(start_date)
    end_year = get_year(end_date)
    start_month = get_month(start_date)
    end_month = get_month(end_date)
    start_day = np.maximum(np.minimum(get_day(start_date), 30), 30 * is_eom(start_date))
    end_day = np.maximum(np.minimum(get_day(end_date), 30), 30 * is_eom(end_date))
    return (360 * (end_year - start_year) + 30 * (end_month - start_month) + (end_day - start_day)) / 360.0


def thirty_e_360(start_date: np.datetime64, end_date: np.datetime64) -> float:
    """ THIRTY E 360 day count standard."""
    start_year = get_year(start_date)
    end_year = get_year(end_date)
    start_month = get_month(start_date)
    end_month = get_month(end_date)
    start_day = np.minimum(get_day(start_date), 30)
    end_day = np.minimum(get_day(end_date), 30)
    return (360.0 * (end_year - start_year) + 30.0 * (end_month - start_month) + (end_day - start_day)) / 360.0
