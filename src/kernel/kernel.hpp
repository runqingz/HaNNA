#include <string>
#include <vector>
#include <cstdlib>
#include "Halide.h"

namespace Kernel {
    struct Tensor {
        Halide::Func f;
        std::vector<int> shape;
        std::string name;
    };

    struct WeightShape {
        int c;  // output channels
        int w;
        int h;
        int pad;
        int stride;
    };

    Tensor conv2D(const Tensor &input, const WeightShape &weight_shape, const Halide::Func &weights, const std::string &name);
    Tensor fc_layer(const Tensor &input, const WeightShape &weight_shape, const Halide::Func &weights, const Halide::Func &bias, const std::string &name);
    Tensor relu_layer(const Tensor &input, const std::string &name);
    Tensor max_pool_layer(const Tensor &input, const WeightShape &weight_shape, const std::string &name);
    Tensor avg_pool_layer(const Tensor &input, const WeightShape &weight_shape, const std::string &name);
    Tensor norm_layer(const Tensor &input, const Halide::Func &mu, const Halide::Func &sigma, const std::string &name);
    Tensor scale_layer(const Tensor &input, const Halide::Func &gamma, const Halide::Func &beta, const std::string &name);
    Tensor sum_layer(const Tensor &t1, const Tensor &t2, const std::string &name);
    Halide::Func softmax_layer(const Tensor &input, const int classes, const std::string &name);


    Halide::Func pad(Halide::Func f, Halide::Expr width, Halide::Expr height);
    std::vector<int> compute_shape(const Tensor &in, const WeightShape &params);
}