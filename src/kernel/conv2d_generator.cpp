#include "Halide.h"
#include <tuple>
#include "kernel_generator.h"

class Conv2DGenerator : public Halide::Generator<Conv2DGenerator> {
    Input<Buffer<float>> input{"input", 4};
    Input<Buffer<float>> conv1_weights{"conv1_weights", 4};
    Input<Buffer<int32_t>> shape;
    Input<int32_t> wc{"channel"};
    Input<int32_t> ww{"width"};
    Input<int32_t> wh{"height"};
    Input<int32_t> wp{"padded"};
    Input<int32_t> ws{"stride"};

    Output<Buffer<float>> output{"output", 4};

    Var c, i, j;

};