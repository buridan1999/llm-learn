#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using tensor_t = std::vector<double>;

bool nearly_equal(double a, double b, double epsilon = 1e-5)
{
    return std::abs(a - b) < epsilon;
}

bool check_vector_equals(const std::vector<double>& a, const std::vector<double>& b)
{
    for(size_t i = 0; i < a.size(); i++) {

        if (!nearly_equal(a[i], b[i])) {

            return false;
        }
    }

    return true;
}

bool check_tensor_equals(const std::vector<tensor_t>& a, const std::vector<tensor_t>& b)
{
    for(size_t i = 0; i < a.size(); i++) {

        if (!check_vector_equals(a[i], b[i])) {

            return false;
        }
    }

    return true;
}

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

// Dot-Product

double dot_product(
    const std::vector<double>& query,
    const std::vector<double>& key
)
{
    // assert(query.size() == key.size())

    // dot = Q * K (scalar)
    double dot = 0.0;
    for(size_t i = 0; i < query.size(); i++) {

        dot += query[i] * key[i];
    }

    return dot;
}

// Scaled Dot-Product Attention

tensor_t scaled_dot_product_attention(
    const tensor_t query,
    const std::vector<tensor_t>& keys,
    const std::vector<tensor_t>& values
)
{
    const size_t d_k = keys[0].size();
    const double sqrt_d_k = sqrt(static_cast<double>(d_k));

    std::vector<double> priorities(keys.size());

    for(size_t i = 0; i < keys.size(); i++) {

        const auto& key = keys[i];
        priorities[i] = dot_product(query, key);
    }

    // scaled priorities calc
    for(size_t i = 0; i < priorities.size(); i++) {

        priorities[i] = priorities[i] / sqrt_d_k;
    }

    return priorities;
}


tensor_t apply_attention(
    const tensor_t& weights,
    const std::vector<tensor_t>& values
)
{
    const size_t d_k = values[0].size();

    tensor_t result(d_k, 0.0);

    for(size_t i = 0; i < values.size(); i++)
    {
        const double w = weights[i];

        for(size_t dim = 0; dim < d_k; dim++)
        {
            result[dim] += w * values[i][dim];
        }
    }

    return result;
}

std::vector<tensor_t> scaled_dot_product_attention(
    const std::vector<tensor_t>& queries,
    const std::vector<tensor_t>& keys,
    const std::vector<tensor_t>& values
)
{
    const size_t d_k = keys[0].size();

    std::vector<tensor_t> outputs(queries.size());

    for(size_t i = 0; i < queries.size(); i++) {

        // process each query
        const auto& query = queries[i];

        // todo: call scaled_dot_product_attention
        const auto scores
            = scaled_dot_product_attention(query, keys, values);

        const auto weights = softmax2(scores);

        outputs[i] = apply_attention(weights, values);
    }

    return outputs;
}

int main()
{
    std::cout << "Hello, World!\n";

    const std::vector<tensor_t> queries = {
        {1.0, 2.0}
    };

    const std::vector<tensor_t> keys = {
        {1.0, 2.0},
        {2.0, 1.0},
        {0.0, 1.0},
        {1.0, 0.0}
    };

    const std::vector<tensor_t> values = {
        {10.0, 0.0},
        {0.0, 10.0},
        {5.0, 5.0},
        {1.0, 1.0}
    };

    // result must be:
    const std::vector<tensor_t> attention = {
        {6.3745, 3.3427}
    };


    const auto result = scaled_dot_product_attention(queries, keys, values);
    print_vector(result[0]);

    std::cout << "attention(test,calc) equal: " << (check_tensor_equals(attention,result) ? "true":"false") << std::endl;

    return 0;
}
