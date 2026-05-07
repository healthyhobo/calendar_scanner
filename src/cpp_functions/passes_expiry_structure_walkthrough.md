# Exercise: `passes_expiry_structure` in C++

This exercise translates `passes_expiry_structure` from `src/signals.py`.

## Why This Rule Exists

A calendar spread is a time spread: it sells one expiry and buys a later expiry at roughly the same strike. That means the back option must expire after the front option.

The rule protects the strategy from accidental structures that are not really calendars:

- The front option must not be too close to expiry.
- The back option must be farther out than the front option.
- The gap between expiries must be large enough.
- Optional min/max DTE limits keep trades in the intended expiry zone.

This matters because short-dated options have very high gamma near expiry. A front option with only a few days left can behave very differently from a 20-30 DTE front leg.

## Key Terms

`DTE` means days to expiration.

For a long calendar:

```text
front_dte = days until the short option expires
back_dte  = days until the long option expires
```

For a clean calendar:

```text
back_dte > front_dte
```

The project also enforces a hard minimum of 10 days for the front leg, even if the config says something lower.

## Translation Map

| Python | C++ |
|---|---|
| `expiry.get("front_dte_min", 10)` | `cfg.front_dte_min` |
| `max(a, b)` | `std::max(a, b)` |
| `None` or `np.nan` optional value | `NaN` in this exercise |
| `np.isnan(x)` | `std::isnan(x)` |
| `return False` | `return false` |

## Step-by-Step Plan

1. Open `passes_expiry_structure_exercise.cpp`.
2. Reject missing `front_dte`.
3. Calculate `front_min = std::max(cfg.front_dte_min, 10.0)`.
4. Reject `front_dte < front_min`.
5. Apply `front_dte_max` if it is not missing.
6. If `back_dte` is present, validate the back leg.
7. Return `true` if all rules pass.

## Compile and Run

```powershell
g++ -std=c++17 src/cpp_functions/passes_expiry_structure_exercise.cpp -o passes_expiry_structure_exercise
.\passes_expiry_structure_exercise.exe
```

Or with Microsoft `cl`:

```powershell
cl /EHsc /std:c++17 src\cpp_functions\passes_expiry_structure_exercise.cpp
.\passes_expiry_structure_exercise.exe
```

## Check Yourself

Expected behavior:

```text
valid calendar -> true
front too short -> false
back before front -> false
gap too small -> false
missing back dte is allowed -> true
```

## References

- OCC options education on options expiration and contract basics: https://www.optionseducation.org/optionsoverview/options-contracts
- Cboe Options Institute glossary for expiration date and calendar spread terms: https://www.cboe.com/optionsinstitute/glossary/
