/*
Exercise: translate Python check_exit into C++.

Python source:
    src/signals.py -> check_exit

Goal:
    Complete the TODOs so this function returns the correct exit reason
    for an open calendar-spread position.

This exercise uses hold_days directly instead of date objects. That keeps
the lesson focused on the exit rules rather than C++ date libraries.
*/

#include <cmath>
#include <iostream>
#include <optional>
#include <string>

enum class ExitReason {
    Normalization,
    ProfitTarget,
    StopLoss,
    TimeStop,
    MaxHold
};

std::string to_string(ExitReason reason) {
    switch (reason) {
        case ExitReason::Normalization: return "normalization";
        case ExitReason::ProfitTarget:  return "profit_target";
        case ExitReason::StopLoss:      return "stop_loss";
        case ExitReason::TimeStop:      return "time_stop";
        case ExitReason::MaxHold:       return "max_hold";
    }
    return "unknown";
}

struct ExitConfig {
    double stop_loss_pct;
    double profit_target_pct;
    int min_hold_days;
    double spread_compression_ratio;      // NaN means disabled
    double normalization_min_pnl_pct;
    double time_stop_dte;
    int max_hold_days;
};

struct SignalExitConfig {
    double zscore_exit;
};

struct ExitRow {
    int hold_days;
    double current_pnl_pct;
    double current_zscore;
    double current_front_dte;
    double entry_iv_spread;
    double current_front_iv;
    double current_back_iv;
};

bool is_missing(double value) {
    return std::isnan(value);
}

std::optional<ExitReason> check_exit(
    const ExitRow& row,
    const ExitConfig& exit_cfg,
    const SignalExitConfig& sig_cfg
) {
    /*
    Python exit priority:
        1. stop loss
        2. profit target
        3. if hold_days >= min_hold_days:
             - normalization
             - time stop
        4. max hold
        5. no exit
    */

    /*
    TODO 1:
        Risk exits fire immediately.

        If current_pnl_pct <= stop_loss_pct, return StopLoss.
        If current_pnl_pct >= profit_target_pct, return ProfitTarget.
    */

    /*
    TODO 2:
        Only check normalization and time stop after min_hold_days.

        Hint:
            if (row.hold_days >= exit_cfg.min_hold_days) { ... }
    */

    /*
    TODO 3:
        Inside the min-hold block, calculate global_z_ok:

            current_zscore is present
            and current_zscore <= zscore_exit
    */

    /*
    TODO 4:
        Calculate current_spread if both front and back IV are present:

            current_spread = current_front_iv - current_back_iv

        If either is missing, use NaN.
    */

    /*
    TODO 5:
        Calculate spread_compressed.

        If spread_compression_ratio is NaN, treat compression as true.
        Otherwise require:
            entry_iv_spread is present
            current_spread is present
            current_spread <= entry_iv_spread * spread_compression_ratio
    */

    /*
    TODO 6:
        If global_z_ok, spread_compressed, and current_pnl_pct is at least
        normalization_min_pnl_pct, return Normalization.
    */

    /*
    TODO 7:
        Still inside the min-hold block, apply time stop:

            if current_front_dte is present and current_front_dte < time_stop_dte,
            return TimeStop.
    */

    /*
    TODO 8:
        Max hold fires after the min-hold block:

            if hold_days >= max_hold_days, return MaxHold.
    */

    /*
    TODO 9:
        If no exit rule fired, return std::nullopt.
    */
    return std::nullopt;
}

std::string show(std::optional<ExitReason> reason) {
    if (!reason.has_value()) {
        return "none";
    }
    return to_string(reason.value());
}

void print_case(const std::string& label, std::optional<ExitReason> actual, const std::string& expected) {
    std::cout << label << " -> actual=" << show(actual)
              << ", expected=" << expected << '\n';
}

int main() {
    const double nan = std::nan("");
    const ExitConfig exit_cfg{
        -0.30,  // stop_loss_pct
        0.40,   // profit_target_pct
        3,      // min_hold_days
        0.40,   // spread_compression_ratio
        0.00,   // normalization_min_pnl_pct
        5.0,    // time_stop_dte
        30      // max_hold_days
    };
    const SignalExitConfig sig_cfg{-0.25};

    print_case("stop loss has priority",
               check_exit({1, -0.35, 2.0, 20.0, 2.0, 22.0, 20.0}, exit_cfg, sig_cfg),
               "stop_loss");

    print_case("profit target has priority",
               check_exit({1, 0.45, 2.0, 20.0, 2.0, 22.0, 20.0}, exit_cfg, sig_cfg),
               "profit_target");

    print_case("normalization after min hold",
               check_exit({5, 0.10, -0.40, 20.0, 2.0, 20.5, 20.0}, exit_cfg, sig_cfg),
               "normalization");

    print_case("time stop after min hold",
               check_exit({5, -0.05, 1.0, 3.0, 2.0, 22.0, 20.0}, exit_cfg, sig_cfg),
               "time_stop");

    print_case("max hold",
               check_exit({35, -0.05, 1.0, 10.0, 2.0, 22.0, 20.0}, exit_cfg, sig_cfg),
               "max_hold");

    print_case("no exit yet",
               check_exit({2, -0.05, -0.50, 20.0, 2.0, 20.5, 20.0}, exit_cfg, sig_cfg),
               "none");

    print_case("disabled compression ratio allows z-score exit",
               check_exit({5, 0.02, -0.40, 20.0, nan, nan, nan},
                          {-0.30, 0.40, 3, nan, 0.00, 5.0, 30},
                          sig_cfg),
               "normalization");

    return 0;
}
