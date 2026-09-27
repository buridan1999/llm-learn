#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// example:
// input: [2.0, 1.0, 0.0]
// result: ~[0.66524096, 0.24472847, 0.09003057]
//
// impl:
// softmax(xi​)=∑j​exj​exi​​
std::vector<double> softmax(const std::vector<double>& input)
{
    std::vector<double> softmax_values(input.size());

    double ex_sum = 0.0;
    for(size_t i = 0; i < input.size(); i++) {

        softmax_values[i] = std::exp(input[i]); // save all exponents calcs
        ex_sum += softmax_values[i];
    }

    for(size_t i = 0; i < input.size(); i++) {

        softmax_values[i] = softmax_values[i] / ex_sum;
    }

    return softmax_values;
}

// replace exp(x) with exp(x - max)
std::vector<double> softmax2(const std::vector<double>& input)
{
    const double max_value = *std::max_element(input.cbegin(), input.cend());

    std::vector<double> softmax_values(input.size());

    double ex_sum = 0.0;
    for(size_t i = 0; i < input.size(); i++) {

        softmax_values[i] = std::exp(input[i] - max_value); // save all exponents calcs
        ex_sum += softmax_values[i];
    }

    for(size_t i = 0; i < input.size(); i++) {

        softmax_values[i] = softmax_values[i] / ex_sum;
    }

    return softmax_values;

}

void print_vector(const std::vector<double>& input)
{
    for(const auto it : input) {

        std::cout << it << " ";
    }
    std::cout << std::endl;
}

bool check_x0_more_x1_then_sm0_more_sm1(const std::vector<double>& x_vector, const std::vector<double>& sm_vector)
{
    for(size_t i = 1; i < x_vector.size(); i++) {

        const auto x1 = x_vector[i];
        const auto x0 = x_vector[i - 1];
        const bool x0_greater_x1 = x0 > x1;

        const auto sm1 = sm_vector[i];
        const auto sm0 = sm_vector[i - 1];
        const bool sm0_greater_sm1 = sm0 > sm1;

        if (x0_greater_x1 != sm0_greater_sm1) {

            return false;
        }
    }

    return true;
}

int main()
{
    std::cout << "Hello, World!\n";

    const std::vector<double> inputs[] = {
        {3.0, 2.0, 1.0, 0.0},
        {1000.0, 1001.0, 1002.0, 1003.0},
        {0.0, 0.0, 0.0, 0.0},
        {10.0, 1.0, 0.0},
        {-5.0, -4.0, -3.0},
    };

    for(const auto& input : inputs) {

        std::cout << "input: ";
        print_vector(input);

        const auto res_a = softmax(input);
        print_vector(res_a);

        const auto res_b = softmax2(input);
        print_vector(res_b);

        const bool x0sm0equation = check_x0_more_x1_then_sm0_more_sm1(input, res_b);
        std::cout << "x0sm0equation: " << (x0sm0equation ? "true" : "false") << std::endl;

    }

    return 0;
}
