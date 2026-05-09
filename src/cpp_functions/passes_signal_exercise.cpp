/*
Exercise: translate Python passes_signal into C++.

Python source:
    src/signals.py -> passes_signal

Goal:
    Complete the TODOs so this function checks the term-structure signal
    filters used before entering a trade.
*/

#include <cmath>
#include <iostream>
#include <string>

struct SignalConfig {
    double zscore_entry_min;
    double zscore_entry_max;
    double percentile_entry_min;
    double back_iv_percentile_max;        // NaN means disabled
    double rv_iv_ratio_max;               // NaN means disabled
    double max_gamma_drag_pct_of_debit;   // NaN means disabled
};

struct SignalRow {
    double spread_zscore;
    double spread_pctile;
    double back_iv_pctile;
    double rv_iv_ratio;
    double daily_gamma_drag;
    double calendar_debit;
};

bool is_missing(double value) {
    return std::isnan(value);
}

bool passes_signal(const SignalRow& row, const SignalConfig& cfg) {
    /*
    Python logic summary:
        - spread_zscore and spread_pctile must exist
        - zscore must be inside [entry_min, entry_max]
        - percentile must be above entry minimum
        - optional back IV percentile cap
        - optional RV/IV cap
        - optional gamma drag as fraction of calendar debit cap
    */

    /*
    TODO 1:
        Reject missing z-score or missing percentile.
    */
   if (is_missing(row.spread_zscore) || is_missing(row.spread_pctile)) {
        return false;
    }

    /*
    TODO 2:
        Apply the z-score band:
            zscore must be >= cfg.zscore_entry_min
            zscore must be <= cfg.zscore_entry_max
    */
   if (!(row.spread_zscore >= cfg.zscore_entry_min && row.spread_zscore <= cfg.zscore_entry_max)) {
        return false;
    }

    /*
    TODO 3:
        Apply percentile_entry_min.
    */
    if (row.spread_pctile < cfg.percentile_entry_min) {
        return false;
    }

    /*
    TODO 4:
        If back_iv_percentile_max is enabled and row.back_iv_pctile is present,
        reject rows where back_iv_pctile is above the cap.

    Hint:
        The cap is enabled when !is_missing(cfg.back_iv_percentile_max).
    */
   if (!is_missing(cfg.back_iv_percentile_max) && !is_missing(row.back_iv_pctile) && row.back_iv_pctile > cfg.back_iv_percentile_max) {
        return false;
   }

    /*
    TODO 5:
        If rv_iv_ratio_max is enabled and row.rv_iv_ratio is present,
        reject rows where rv_iv_ratio is above the cap.
    */
   if (!is_missing(cfg.rv_iv_ratio_max) && !is_missing(row.rv_iv_ratio) && row.rv_iv_ratio > cfg.rv_iv_ratio_max) {
        return false;
   }

    /*
    TODO 6:
        If max_gamma_drag_pct_of_debit is enabled and both gamma drag and
        debit are present, reject rows where:

            daily_gamma_drag / calendar_debit > max_gamma_drag_pct_of_debit

        Only apply this when calendar_debit > 0.
    */
   if (!is_missing(cfg.max_gamma_drag_pct_of_debit) && !is_missing(row.daily_gamma_drag) && !is_missing(row.calendar_debit) && row.calendar_debit > 0) {
        double gamma_drag_pct_of_debit = row.daily_gamma_drag / row.calendar_debit;
        if (gamma_drag_pct_of_debit > cfg.max_gamma_drag_pct_of_debit) {
            return false;
        }
    }

    /*
    TODO 7:
        Return true if all checks passed.
    */
    return true;
}

void print_case(const std::string& label, bool actual, bool expected) {
    std::cout << label << " -> actual=" << std::boolalpha << actual
              << ", expected=" << expected << '\n';
}

int main() {
    const double nan = std::nan("");
    const SignalConfig cfg{
        1.5,    // zscore_entry_min
        4.0,    // zscore_entry_max
        80.0,   // percentile_entry_min
        60.0,   // back_iv_percentile_max
        0.9,    // rv_iv_ratio_max
        0.15    // max_gamma_drag_pct_of_debit
    };

    print_case("valid signal",
               passes_signal({2.2, 90.0, 50.0, 0.7, 0.10, 1.00}, cfg),
               true);

    print_case("zscore too low",
               passes_signal({1.1, 90.0, 50.0, 0.7, 0.10, 1.00}, cfg),
               false);

    print_case("percentile too low",
               passes_signal({2.2, 70.0, 50.0, 0.7, 0.10, 1.00}, cfg),
               false);

    print_case("back iv too elevated",
               passes_signal({2.2, 90.0, 75.0, 0.7, 0.10, 1.00}, cfg),
               false);

    print_case("rv iv too high",
               passes_signal({2.2, 90.0, 50.0, 1.1, 0.10, 1.00}, cfg),
               false);

    print_case("gamma drag too high",
               passes_signal({2.2, 90.0, 50.0, 0.7, 0.20, 1.00}, cfg),
               false);

    print_case("missing optional filters are allowed",
               passes_signal({2.2, 90.0, nan, nan, nan, nan}, cfg),
               true);

    return 0;
}
