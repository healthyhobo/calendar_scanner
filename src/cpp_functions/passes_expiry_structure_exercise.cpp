/*
Exercise: translate Python passes_expiry_structure into C++.

Python source:
    src/signals.py -> passes_expiry_structure

Goal:
    Complete the TODOs so this function confirms that a row represents a
    valid calendar-spread expiry structure.
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

struct ExpiryConfig {
    double front_dte_min;
    double front_dte_max;  // NaN means disabled
    double back_dte_min;   // NaN means disabled
    double back_dte_max;   // NaN means disabled
    double min_gap_days;   // NaN means disabled
};

struct ExpiryRow {
    double front_dte;
    double back_dte;       // NaN means unavailable
};

bool is_missing(double value) {
    return std::isnan(value);
}

bool passes_expiry_structure(const ExpiryRow& row, const ExpiryConfig& cfg) {
    /*
    Python logic summary:
        - front_dte must exist
        - front_dte must be >= max(configured front min, 10)
        - if configured, front_dte must be <= front max
        - if back_dte exists:
            - back_dte must be greater than front_dte
            - if configured, back_dte must be >= back min
            - if configured, back_dte must be <= back max
            - if configured, back_dte - front_dte must be >= min gap
    */

    /*
    TODO 1:
        Reject missing front_dte.
    */

    /*
    TODO 2:
        Calculate the hard front minimum.

        Python:
            front_min = max(int(expiry.get("front_dte_min", 10)), 10)

        C++ hint:
            double front_min = std::max(cfg.front_dte_min, 10.0);
    */

    /*
    TODO 3:
        Reject rows where front_dte is below front_min.
    */

    /*
    TODO 4:
        If front_dte_max is not missing, reject rows above it.

    Hint:
        if (!is_missing(cfg.front_dte_max) && row.front_dte > cfg.front_dte_max) ...
    */

    /*
    TODO 5:
        If back_dte is present, validate the back leg:
            - back_dte must be greater than front_dte
            - back min applies if configured
            - back max applies if configured
            - min gap applies if configured
    */

    /*
    TODO 6:
        Return true if all checks passed.
    */
    return false;
}

void print_case(const std::string& label, bool actual, bool expected) {
    std::cout << label << " -> actual=" << std::boolalpha << actual
              << ", expected=" << expected << '\n';
}

int main() {
    const double nan = std::nan("");
    const ExpiryConfig cfg{
        7.0,    // front_dte_min, but hard floor is 10
        45.0,   // front_dte_max
        30.0,   // back_dte_min
        90.0,   // back_dte_max
        14.0    // min_gap_days
    };

    print_case("valid calendar",
               passes_expiry_structure({25.0, 55.0}, cfg),
               true);

    print_case("front too short",
               passes_expiry_structure({8.0, 40.0}, cfg),
               false);

    print_case("back before front",
               passes_expiry_structure({30.0, 25.0}, cfg),
               false);

    print_case("gap too small",
               passes_expiry_structure({25.0, 34.0}, cfg),
               false);

    print_case("missing back dte is allowed",
               passes_expiry_structure({25.0, nan}, cfg),
               true);

    return 0;
}
