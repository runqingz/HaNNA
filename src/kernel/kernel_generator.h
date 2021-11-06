#include "Halide.h"

namespace {
    struct WeightShape {
        int c;  // output channels
        int w;
        int h;
        int pad;
        int stride;
    };

    std::vector<int> compute_shape(const WeightShape &in, const WeightShape &params) {
        int w = (1.0 / params.stride) * (params.pad * 2 + in.w - params.w + 1 + params.stride - 1);
        int h = (1.0 / params.stride) * (params.pad * 2 + in.h - params.h + 1 + params.stride - 1);
        int c = params.c;

        return {c, w, h};
    }

    Halide::Func pad(Halide::Func f, Halide::Expr width, Halide::Expr height) {
        Halide::Region bounds(f.dimensions());
        bounds[1].min = 0;
        bounds[1].extent = width;
        bounds[2].min = 0;
        bounds[2].extent = height;
        return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
    }
}
