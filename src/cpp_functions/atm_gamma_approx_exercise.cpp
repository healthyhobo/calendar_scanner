/*
Exercise: translate Python _atm_gamma_approx into C++.

Python source:
    src/features.py -> _atm_gamma_approx

Goal:
    Fill in the TODO sections below so this C++ function matches the
    Python ATM gamma approximation helper.

How to compile from the project root:
    g++ -std=c++17 src/cpp_functions/atm_gamma_approx_exercise.cpp -o atm_gamma_approx_exercise

How to run:
    ./atm_gamma_approx_exercise

On Windows PowerShell, if the executable is created in the current folder:
    .\atm_gamma_approx_exercise.exe
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

double atm_gamma_approx(double spot, double iv_dec, double dte) {
    /*
    TODO 1:
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
    TODO 2:
        Translate:

            T = max(dte, 1) / 365.25

    Hint:
        We included <algorithm>, so you can use:
            std::max(dte, 1.0)

        Make sure both values are doubles.
    */
   double T = std::max(dte, 1.0) / 365.25;

    /*
    TODO 3:
        Return the ATM gamma approximation:

            0.3989 / (spot * iv_dec * sqrt(T))

    Hint:
        Use std::sqrt(T).

        0.3989 is approximately N'(0), the height of the standard normal
        density at zero. For ATM options, d1 is often close to zero, so
        this is a useful quick approximation.
    */
   return 0.3989 / (spot * iv_dec * std::sqrt(T));
}

void print_case(double spot, double iv_dec, double dte, double expected) {
    const double actual = atm_gamma_approx(spot, iv_dec, dte);
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "spot=" << spot
              << ", iv_dec=" << iv_dec
              << ", dte=" << dte
              << " -> gamma=" << actual
              << " | expected about " << expected << '\n';
}

int main() {
    std::cout << "ATM gamma approximation exercise\n";
    std::cout << "If you still see nan, complete the TODOs.\n\n";

    print_case(100.0, 0.20, 30.0, 0.06959348);
    print_case(50.0, 0.35, 45.0, 0.06494038);
    print_case(425.0, 0.18, 60.0, 0.01286536);

    return 0;
}