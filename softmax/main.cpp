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

void print_vector(const std::vector<double>& input)
{
    for(const auto it : input) {

        std::cout << it << " ";
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "Hello, World!\n";

    const auto res_a = softmax({2.0, 1.0, 0.0});
    print_vector(res_a);

    const auto res_b = softmax({1000.0, 1001.0, 1002.0});
    print_vector(res_b);

    return 0;
}
