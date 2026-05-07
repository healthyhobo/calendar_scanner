# Exercise: `check_exit` in C++

This exercise translates the exit-rule logic from `src/signals.py`.

The Python function returns one of these strings, or `None`:

```python
class ExitReason:
    NORMALIZATION = "normalization"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS     = "stop_loss"
    TIME_STOP     = "time_stop"
    MAX_HOLD      = "max_hold"
```

The C++ exercise uses:

```cpp
enum class ExitReason
std::optional<ExitReason>
```

`std::optional` is how this exercise represents "maybe there is an exit reason, maybe there is not."

## Why This Rule Exists

Entry rules decide when to open a trade. Exit rules decide when the trade thesis has ended or when risk must be controlled.

For this calendar-spread strategy, there are five exits:

- `stop_loss`: the trade lost too much.
- `profit_target`: the trade hit a predefined gain.
- `normalization`: the volatility spread mean-reverted.
- `time_stop`: the front option is getting too close to expiry.
- `max_hold`: the trade has been open too long.

The order matters. Stop loss and profit target fire immediately, even if the trade was just opened. Normalization and time stop wait until `min_hold_days`, which helps avoid exiting from short-term noise.

## Exit Priority

The Python function checks exits in this order:

```text
1. stop loss
2. profit target
3. normalization, only after min_hold_days
4. time stop, only after min_hold_days
5. max hold
6. no exit
```

Priority matters. If a trade is both below the stop loss and past max hold, the function returns `stop_loss`, because risk exits are checked first.

## Trading Intuition

The main desired exit is normalization:

```text
entry spread was unusually wide
current spread has compressed
z-score has fallen below the exit threshold
P&L is not below the minimum allowed normalization P&L
```

The time stop protects against the front leg becoming too close to expiry. Near expiry, gamma risk can rise quickly, and assignment/pin risk can become more relevant.

The max hold rule is a final safety cap. Even if the signal never normalizes, the strategy does not hold forever.

## Translation Map

| Python | C++ |
|---|---|
| string constants | `enum class ExitReason` |
| `None` | `std::nullopt` |
| `str | None` | `std::optional<ExitReason>` |
| `np.isnan(x)` / `_is_nan(x)` | `std::isnan(x)` |
| `and` | `&&` |
| `or` | `||` |

## Step-by-Step Plan

1. Open `check_exit_exercise.cpp`.
2. Complete the stop-loss and profit-target checks.
3. Add the `if (row.hold_days >= exit_cfg.min_hold_days)` block.
4. Inside that block, calculate `global_z_ok`.
5. Calculate `current_spread`.
6. Calculate `spread_compressed`.
7. Return `Normalization` when all normalization conditions are true.
8. Add the time-stop rule.
9. Add the max-hold rule.
10. Return `std::nullopt` when there is no exit.

## Compile and Run

```powershell
g++ -std=c++17 src/cpp_functions/check_exit_exercise.cpp -o check_exit_exercise
.\check_exit_exercise.exe
```

Or with Microsoft `cl`:

```powershell
cl /EHsc /std:c++17 src\cpp_functions\check_exit_exercise.cpp
.\check_exit_exercise.exe
```

## Check Yourself

Expected behavior:

```text
stop loss has priority -> stop_loss
profit target has priority -> profit_target
normalization after min hold -> normalization
time stop after min hold -> time_stop
max hold -> max_hold
no exit yet -> none
disabled compression ratio allows z-score exit -> normalization
```

## C++ Concept: `enum class`

An `enum class` gives a fixed set of named values:

```cpp
enum class ExitReason {
    Normalization,
    ProfitTarget,
    StopLoss,
    TimeStop,
    MaxHold
};
```

This is safer than passing raw strings everywhere. The compiler can help catch typos.

## C++ Concept: `std::optional`

`std::optional<ExitReason>` means:

```text
there may be an ExitReason
or there may be nothing
```

That mirrors Python's:

```python
return None
```

## References

- C++ reference for `std::optional`: https://en.cppreference.com/w/cpp/utility/optional
- C++ reference for scoped enumerations: https://en.cppreference.com/w/cpp/language/enum
- Cboe Options Institute glossary for expiration, assignment, and Greeks: https://www.cboe.com/optionsinstitute/glossary/
