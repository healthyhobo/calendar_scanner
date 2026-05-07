/*
Exercise: translate Python _bs_atm_call into C++.

Python source:
    src/features.py -> _bs_atm_call

Goal:
    Fill in the TODO sections below so this C++ function matches the
    Python Black-Scholes ATM call helper.

How to compile from the project root:
    g++ -std=c++17 src/cpp_functions/bs_atm_call_exercise.cpp -o bs_atm_call_exercise

How to run:
    ./bs_atm_call_exercise

On Windows PowerShell, if the executable is created in the current folder:
    .\bs_atm_call_exercise.exe
*/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

double normal_cdf(double x) {
    /*
    TODO 1:
        Implement the standard normal cumulative distribution function N(x).

    Hint:
        C++ has std::erfc in <cmath>.

        A common identity is:
            N(x) = 0.5 * erfc(-x / sqrt(2))

    C++ ingredients:
        std::erfc(...)
        std::sqrt(...)
    */
    return 0.5 * std::erfc(-x / std::sqrt(2));
}

double bs_atm_call(double spot, double iv_dec, double dte, double r = 0.04) {
    /*
    TODO 2:
        Translate the Python guard clause:

            if dte <= 0 or iv_dec <= 0 or spot <= 0:
                return np.nan

    Hint:
        In C++, use || for "or".
        To return NaN:
            return std::numeric_limits<double>::quiet_NaN();
    */
   if(dte <= 0 || iv_dec <=0 || spot <= 0)
   {
    return std::numeric_limits<double>::quiet_NaN();
   }

    /*
    TODO 3:
        Convert days to years:

            T = dte / 365.25

        Then compute:

            sqrt_T = sqrt(T)

    Hint:
        Use double variables.
    */
   double T = dte / 365.25;
   double sqrt_T = std::sqrt(T);

    /*
    TODO 4:
        Translate d1 and d2 from Python:

            d1 = (r + 0.5 * iv_dec ** 2) * T / (iv_dec * sqrt_T)
            d2 = d1 - iv_dec * sqrt_T

    Hint:
        C++ does not use ** for powers.
        For iv_dec squared, either use:
            iv_dec * iv_dec
        or:
            std::pow(iv_dec, 2.0)
    */
   double d1 = (r + 0.5 * std::pow(iv_dec, 2.0)) * T / (iv_dec * sqrt_T);
   double d2 = d1 - iv_dec * sqrt_T;

    /*
    TODO 5:
        Return the ATM call price:

            spot * N(d1) - spot * exp(-r * T) * N(d2)

    Hint:
        Use normal_cdf(d1), normal_cdf(d2), and std::exp(...).
        Because this function is ATM, strike K equals spot S, so the
        usual K term becomes spot.
    */
    return spot * normal_cdf(d1) - spot * std::exp(-r * T) * normal_cdf(d2);
}

void print_case(double spot, double iv_dec, double dte, double r, double expected) {
    const double actual = bs_atm_call(spot, iv_dec, dte, r);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "spot=" << spot
              << ", iv_dec=" << iv_dec
              << ", dte=" << dte
              << ", r=" << r
              << " -> price=" << actual
              << " | expected about " << expected << '\n';
}

int main() {
    std::cout << "Black-Scholes ATM call exercise\n";
    std::cout << "If you still see nan, complete the TODOs.\n\n";

    print_case(100.0, 0.20, 30.0, 0.04, 2.450367);
    print_case(50.0, 0.35, 45.0, 0.04, 2.567829);
    print_case(425.0, 0.18, 60.0, 0.04, 13.767923);

    return 0;
}