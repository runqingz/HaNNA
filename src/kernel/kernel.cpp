#include "kernel.hpp"
using namespace Halide;

namespace Kernel{
    Var c, i, j;

    Tensor conv2D(const Tensor &input, const WeightShape &weight_shape, const Func &weights, const std::string &name) {
        int p = weight_shape.pad;
        Func padded;
        // pad input
        if (p) {
            padded = pad(input.f, input.shape[1], input.shape[2]);
        } else {
            padded = input.f;
        }
        RDom r(0, input.shape[0], 0, weight_shape.w, 0, weight_shape.h);
        Func conv;
        conv(c, i, j) += weights(c, r.y, r.z, r.x) * padded(r.x, weight_shape.stride * i + r.y - p, weight_shape.stride * j + r.z - p);

        Tensor output;

        //TODO: schedule this
        output.f = conv;
        output.name = name;
        output.shape = compute_shape(input, weight_shape);
        return output;
    }
}