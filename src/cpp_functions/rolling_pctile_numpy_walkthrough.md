# Exercise: `_rolling_pctile_numpy` in C++

This exercise translates the rolling percentile idea from `src/features.py`.

The pure NumPy fallback currently looks like this:

```python
def _rolling_pctile_numpy(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        w = arr[start : i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) < min_periods:
            continue
        cur = valid[-1]
        above = int(np.sum(cur >= valid[:-1]))
        denom = len(valid) - 1
        result[i] = above / denom * 100.0 if denom > 0 else np.nan
    return result
```

The project also has a faster numba path above it. The C++ exercise follows the intended trailing-percentile behavior from that accelerated path: rank the current value against valid prior values in the rolling window. If the current value is missing, the result is missing.

## Why This Function Exists

Your strategy wants to know whether today's IV spread is unusual compared with that ticker's own recent history.

For example, suppose the recent `iv_spread` values are:

```text
1.0, 2.0, 3.0
```

At the third value, `3.0` is greater than both prior values, so its rolling percentile is:

```text
2 / 2 * 100 = 100
```

If the current value is `2.0` and the prior values are `1.0, 2.0, 3.0`, then:

```text
2.0 >= 1.0  yes
2.0 >= 2.0  yes
2.0 >= 3.0  no
```

So the percentile is:

```text
2 / 3 * 100 = 66.6667
```

This is useful because a raw spread like `1.5` may be extreme for one ticker and normal for another. Rolling percentiles make the signal ticker-relative.

## Why "Trailing" Matters

The function only compares today to past values, not future values. That avoids look-ahead bias.

In a backtest, look-ahead bias means accidentally using information that would not have been available at the time of the trade. A rolling trailing percentile is safer because it asks:

```text
As of today, how unusual is today's value compared with recent history?
```

not:

```text
How unusual is today's value compared with the whole future dataset?
```

## Important Inputs

`window` controls how far back the function looks.

```text
window = 252
```

means about one trading year of history.

`min_periods` controls how much history is required before producing a result.

```text
min_periods = 60
```

means the function leaves early rows as `NaN` until it has enough valid observations.

## Translation Map

| Python | C++ |
|---|---|
| `np.ndarray` | `std::vector<double>` |
| `len(arr)` | `values.size()` |
| `np.full(n, np.nan)` | `std::vector<double>(n, nan)` |
| `range(a, b)` | `for (int i = a; i < b; ++i)` |
| `max(0, x)` | `std::max(0, x)` |
| `np.isnan(x)` | `std::isnan(x)` |
| `continue` | `continue` |

## Step-by-Step Plan

1. Open `rolling_pctile_numpy_exercise.cpp`.
2. Add guards for invalid `window` or `min_periods`.
3. Write the outer loop over `i`.
4. Compute `start = std::max(0, i - window + 1)`.
5. Read `cur = values[i]`.
6. Skip the row if `cur` is `NaN`.
7. Loop from `start` to `i - 1`.
8. Count valid prior values.
9. Count how many prior values are less than or equal to the current value.
10. If `count + 1 >= min_periods`, write `above / count * 100.0`.

## Compile and Run

With `g++`:

```powershell
g++ -std=c++17 src/cpp_functions/rolling_pctile_numpy_exercise.cpp -o rolling_pctile_numpy_exercise
.\rolling_pctile_numpy_exercise.exe
```

With Microsoft `cl` from a Developer PowerShell:

```powershell
cl /EHsc /std:c++17 src\cpp_functions\rolling_pctile_numpy_exercise.cpp
.\rolling_pctile_numpy_exercise.exe
```

## Check Yourself

The exercise uses:

```text
values      = 1, 2, 3, 2, 5, nan, 4, 6
window      = 4
min_periods = 3
```

Expected output:

```text
nan nan 100.000000 66.666667 100.000000 nan 50.000000 100.000000
```

Walk through index `3` by hand:

```text
current = 2
window values = 1, 2, 3, 2
prior values = 1, 2, 3
current >= 1 yes
current >= 2 yes
current >= 3 no
percentile = 2 / 3 * 100 = 66.6667
```

Walk through index `6`:

```text
current = 4
window values = 2, 5, nan, 4
prior valid values = 2, 5
current >= 2 yes
current >= 5 no
percentile = 1 / 2 * 100 = 50
```

## Common Beginner Mistakes

- Comparing the current value to itself.
- Forgetting to skip `NaN` prior values.
- Using integer division: `above / count` can become `0` if both are integers. Cast one side to `double`.
- Starting the loop at `0` and producing percentiles before enough history exists.
- Using future values by accidentally looping beyond `i`.

## References

- pandas rolling-window documentation, useful background for the Python version's rolling calculations: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
- NumPy `isnan`, used by the Python code to skip missing values: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
- NumPy `nan`, used as the missing numeric result marker: https://numpy.org/doc/stable/reference/constants.html#numpy.nan
