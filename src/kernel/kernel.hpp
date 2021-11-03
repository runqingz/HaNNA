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

    Halide::Func pad(Halide::Func f, Halide::Expr width, Halide::Expr height);
    std::vector<int> compute_shape(const Tensor &in, const WeightShape &params);
}