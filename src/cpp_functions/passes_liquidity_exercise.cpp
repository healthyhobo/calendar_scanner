/*
Exercise: translate Python passes_liquidity into C++.

Python source:
    src/signals.py -> passes_liquidity

Goal:
    Complete the TODOs so this function accepts liquid rows and rejects
    rows that are too small or too thinly traded.

Compile with g++:
    g++ -std=c++17 src/cpp_functions/passes_liquidity_exercise.cpp -o passes_liquidity_exercise

Compile with Microsoft cl from a Developer PowerShell:
    cl /EHsc /std:c++17 src\cpp_functions\passes_liquidity_exercise.cpp
*/

#include <cmath>
#include <iostream>
#include <string>

struct LiquidityConfig {
    double min_stock_price;
    double min_option_volume;
    double min_avg_opt_volume_20d;
};

struct LiquidityRow {
    double stock_price;
    double total_opt_volume;
    double avgOptVolu20d;
};

bool is_missing(double value) {
    return std::isnan(value);
}

bool passes_liquidity(const LiquidityRow& row, const LiquidityConfig& cfg) {
    /*
    Python logic:

        spot = row.get("stock_price", np.nan)
        if np.isnan(spot) or spot < liq["min_stock_price"]:
            return False

        opt_vol = row.get("total_opt_volume", np.nan)
        avg_vol = row.get("avgOptVolu20d", np.nan)
        avg_floor = liq.get("min_avg_opt_volume_20d", liq["min_option_volume"])

        if not np.isnan(opt_vol) and opt_vol < liq["min_option_volume"]:
            return False
        if not np.isnan(avg_vol) and avg_vol < avg_floor:
            return False

        return True
    */

    /*
    TODO 1:
        Reject missing or too-low stock prices.

    Hint:
        if (is_missing(row.stock_price) || row.stock_price < cfg.min_stock_price) {
            return false;
        }
    */
   if (is_missing(row.stock_price) || row.stock_price < cfg.min_stock_price) {
       return false;
   }

    /*
    TODO 2:
        Reject current option volume only when it is present and too low.

    Hint:
        Missing volume is allowed, just like the Python function.
        Use !is_missing(row.total_opt_volume).
    */
   if (!is_missing(row.total_opt_volume) && row.total_opt_volume < cfg.min_option_volume) {
       return false;
   }

    /*
    TODO 3:
        Reject 20-day average option volume only when it is present and too low.

    Hint:
        Compare avgOptVolu20d to cfg.min_avg_opt_volume_20d.
    */
   if (!is_missing(row.avgOptVolu20d) && row.avgOptVolu20d < cfg.min_avg_opt_volume_20d) {
       return false;
   }

    /*
    TODO 4:
        If none of the rejection rules fired, return true.
    */
    return true;
}

void print_case(const std::string& label, bool actual, bool expected) {
    std::cout << label << " -> actual=" << std::boolalpha << actual
              << ", expected=" << expected << '\n';
}

int main() {
    const double nan = std::nan("");
    const LiquidityConfig cfg{20.0, 1000.0, 2000.0};

    print_case("liquid row",
               passes_liquidity({105.0, 5000.0, 8000.0}, cfg),
               true);

    print_case("stock price too low",
               passes_liquidity({12.0, 5000.0, 8000.0}, cfg),
               false);

    print_case("current option volume too low",
               passes_liquidity({105.0, 500.0, 8000.0}, cfg),
               false);

    print_case("avg option volume too low",
               passes_liquidity({105.0, 5000.0, 1200.0}, cfg),
               false);

    print_case("missing volume fields are allowed",
               passes_liquidity({105.0, nan, nan}, cfg),
               true);

    return 0;
}