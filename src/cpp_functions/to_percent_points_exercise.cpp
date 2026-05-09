/*
Exercise: translate Python _to_percent_points into C++.

Python source:
    src/features.py -> _to_percent_points

Goal:
    Fill in the TODO sections below so this C++ function normalizes
    volatility values into percentage-point form.

How to compile from the project root with g++:
    g++ -std=c++17 src/cpp_functions/to_percent_points_exercise.cpp -o to_percent_points_exercise

How to compile with Microsoft cl from a Developer PowerShell:
    cl /EHsc /std:c++17 src\cpp_functions\to_percent_points_exercise.cpp

How to run from PowerShell:
    .\to_percent_points_exercise.exe
*/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

double to_percent_points(double value) {
    /*
    TODO 1:
        Preserve missing values.

        Python version:
            s = pd.to_numeric(series, errors="coerce")
            finite = s[np.isfinite(s)]

        In this C++ exercise, values are already doubles, so you only need
        to check whether the value is finite.

    Hint:
        Use std::isfinite(value).
        If value is not finite, return it unchanged.
    */
   if (!std::isfinite(value)) {
        return value;
    }

    /*
    TODO 2:
        Translate the threshold rule:

            return s.where(s.abs() > 2.0, s * 100.0)

        Meaning:
            If abs(value) > 2.0, assume it is already percentage points.
            Otherwise, assume it is decimal volatility and multiply by 100.

    Hint:
        Use std::abs(value).
    */
   if (std::abs(value) > 2.0) {
        return value;
    } else {
        return value * 100.0;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> to_percent_points_vector(const std::vector<double>& values) {
    /*
    TODO 3:
        Create an output vector.
        Loop through values.
        Push each converted value into the output vector.
        Return the output vector.

    Hints:
        std::vector<double> output;
        output.push_back(...);

        The single-value helper above should do the actual conversion.
    */
   std::vector<double> output;
   for (double value : values) {
    output.push_back(to_percent_points(value));
   }

    return output;
}

void print_value(double value) {
    if (std::isnan(value)) {
        std::cout << "nan";
    } else {
        std::cout << std::fixed << std::setprecision(4) << value;
    }
}

void print_case(const std::vector<double>& input) {
    const std::vector<double> actual = to_percent_points_vector(input);

    std::cout << "input:  ";
    for (double value : input) {
        print_value(value);
        std::cout << " ";
    }

    std::cout << "\noutput: ";
    for (double value : actual) {
        print_value(value);
        std::cout << " ";
    }

    std::cout << "\nexpected about:\n";
    std::cout << "20.0000 15.0000 150.0000 -30.0000 -25.0000 nan 200.0000 2.0100\n";
}

int main() {
    std::cout << "Volatility percent-point normalization exercise\n";
    std::cout << "If the output is empty or nan-only, complete the TODOs.\n\n";

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> values = {
        0.20,   // decimal 20 percent -> 20.0 percentage points
        15.0,   // already percentage points
        1.50,   // decimal 150 percent -> 150.0 percentage points
        -0.30,  // decimal -30 percent -> -30.0 percentage points
        -25.0,  // already percentage points
        nan,    // missing stays missing
        2.00,   // threshold is > 2.0, so exactly 2.0 becomes 200.0
        2.01    // above threshold, so stays 2.01
    };

    print_case(values);
    return 0;
}
