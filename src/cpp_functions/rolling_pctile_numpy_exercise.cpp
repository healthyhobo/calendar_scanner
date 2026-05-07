/*
Exercise: translate Python _rolling_pctile_numpy into C++.

Python source:
    src/features.py -> _rolling_pctile_numpy

Goal:
    Fill in the TODO sections below so this C++ function computes a
    trailing rolling percentile for each value in a vector.

How to compile from the project root with g++:
    g++ -std=c++17 src/cpp_functions/rolling_pctile_numpy_exercise.cpp -o rolling_pctile_numpy_exercise

How to compile with Microsoft cl from a Developer PowerShell:
    cl /EHsc /std:c++17 src\cpp_functions\rolling_pctile_numpy_exercise.cpp

How to run from PowerShell:
    .\rolling_pctile_numpy_exercise.exe
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

std::vector<double> rolling_pctile_numpy(
    const std::vector<double> &values,
    int window,
    int min_periods)
{
    /*
    This exercise follows the intended trailing-percentile logic:

        For each index i:
            compare values[i] to valid prior values in the rolling window
            percentile = fraction of prior values where current >= prior

    Missing values are represented with NaN.
    */

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int n = static_cast<int>(values.size());
    std::vector<double> result(n, nan);

    /*
    TODO 1:
        Add basic input guards.

        If window <= 0 or min_periods <= 0, return result immediately.

    Hint:
        Use || for "or".
    */
    if (window <= 0 || min_periods <= 0)
    {
        return result;
    }

    /*
    TODO 2:
        Loop over each index i where a result could first appear.

        Python fallback starts at:
            for i in range(min_periods - 1, n):

        Hint:
            for (int i = min_periods - 1; i < n; ++i) { ... }
    */
    for (int i = min_periods - 1; i < n; ++i)
    {

        /*
        TODO 3:
            Inside the loop, compute the rolling window start:

                start = max(0, i - window + 1)

            Hint:
                Use std::max(0, i - window + 1).
        */
        int start = std::max(0, i - window + 1);

        /*
        TODO 4:
            Read the current value:

                cur = values[i]

            If cur is NaN, skip this index and leave result[i] as NaN.

            Hint:
                Use std::isnan(cur).
                Use continue; to skip to the next i.
        */
        if (std::isnan(values[i]))
        {
            continue;
        }
        double cur = values[i];

        /*
        TODO 5:
            Count valid prior observations and how many are <= the current value.

            Python/NumPy idea:
                count = number of valid prior values
                above = number of valid prior values where cur >= prior

            Important:
                Only compare prior values from start through i - 1.
                Do not compare the current value to itself.

            Hints:
                int count = 0;
                int above = 0;

                for (int j = start; j < i; ++j) { ... }

                if prior is NaN, skip it.
                Otherwise increment count.
                If cur >= prior, increment above.
        */
        int count = 0;
        int above = 0;
        for (int j = start; j < i; ++j)
        {
            if (std::isnan(values[j]))
            {
                continue;
            }
            count++;
            if (cur >= values[j])
            {
                above++;
            }
        }

        /*
        TODO 6:
            Only write a percentile if there is enough history.

            Python accelerated path uses:
                if count + 1 >= min_periods:
                    result[i] = above / count * 100.0

            Why count + 1?
                count is prior valid observations.
                +1 includes the current valid observation.

            Also avoid division by zero.
        */
        if (count > 0 && count + 1 >= min_periods)
        {
            result[i] = static_cast<double>(above) / count * 100.0;
        }
    }

    return result;
}

void print_value(double value)
{
    if (std::isnan(value))
    {
        std::cout << "nan";
    }
    else
    {
        std::cout << std::fixed << std::setprecision(6) << value;
    }
}

void print_vector(const std::vector<double> &values)
{
    for (double value : values)
    {
        print_value(value);
        std::cout << " ";
    }
    std::cout << '\n';
}

int main()
{
    std::cout << "Rolling percentile exercise\n";
    std::cout << "If the output is nan-only, complete the TODOs.\n\n";

    const double nan = std::numeric_limits<double>::quiet_NaN();

    const std::vector<double> values = {1.0, 2.0, 3.0, 2.0, 5.0, nan, 4.0, 6.0};
    const std::vector<double> actual = rolling_pctile_numpy(values, 4, 3);

    std::cout << "input:\n";
    print_vector(values);

    std::cout << "\noutput:\n";
    print_vector(actual);

    std::cout << "\nexpected about:\n";
    std::cout << "nan nan 100.000000 66.666667 100.000000 nan 50.000000 100.000000\n";

    return 0;
}
