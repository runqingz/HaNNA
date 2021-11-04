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
    
    // assumes input is 3D (c, w, h) where w and h = 1
    Tensor fc_layer(const Tensor &input, const WeightShape &weight_shape, const Func &weights, const Func &bias, const std::string &name) {
        RDom r(0, input.shape[0]);
        Func fc;
        fc(c) = bias(c);
        fc(c) += weights(c, r.x) * input.f(r.x, 0, 0);

        Tensor output;
        output.f = fc;
        output.name = name;
        output.shape = compute_shape(input, weight_shape);

        return output;
    }

    Tensor relu_layer(const Tensor &input, const std::string &name) {
        Func relu;
        relu(c, i, j) = max(0.0f, input.f(c, i, j));
        Tensor output;
        output.f = relu;
        output.shape = input.shape;
        output.name = name;
        return output;
    }

    Tensor max_pool_layer(const Tensor &input, const WeightShape &weight_shape, const std::string &name) {
        int p = weight_shape.pad;
        Func padded;
        if (p) {
            padded = pad(input.f, input.shape[1], input.shape[2]);
        } else {
            padded = input.f;
        }
        RDom r(0, weight_shape.w, 0, weight_shape.h);
        Func pool;
        pool(c, i, j) = maximum(padded(c, weight_shape.stride * i + r.x - p, weight_shape.stride * j + r.y - p));
        Tensor output;
        output.f = pool;
        output.name = name;
        output.shape = compute_shape(input, weight_shape);

        return output;
    }

    Tensor avg_pool_layer(const Tensor &input, const WeightShape &weight_shape, const std::string &name) {
        int p = weight_shape.pad;
        Func padded;
        if (p) {
            padded = pad(input.f, input.shape[1], input.shape[2]);
        } else {
            padded = input.f;
        }
        RDom r(0, weight_shape.w, 0, weight_shape.h);
        float scale = weight_shape.w * weight_shape.h;
        Func pool;
        float n = 1.0f / scale;
        pool(c, i, j) += n * padded(c, weight_shape.stride * i + r.x - p, weight_shape.stride * j + r.y - p);

        Tensor output;
        output.f = pool;
        output.name = name;
        output.shape = compute_shape(input, weight_shape);

        return output;
    }

    Tensor norm_layer(const Tensor &input, const Func &mu, const Func &sigma, const std::string &name) {
        Func normed;
        Expr e = input.f(c, i, j);
        normed(c, i, j) = (input.f(c, i, j) - mu(c)) / (sqrt(sigma(c) + 1e-5f));
        Tensor output;
        output.f = normed;
        output.shape = input.shape;
        output.name = name;
        return output;
    }

    Tensor scale_layer(const Tensor &input, const Func &gamma, const Func &beta, const std::string &name) {
        Func scaled;
        scaled(c, i, j) = input.f(c, i, j) * gamma(c) + beta(c);
        Tensor output;
        output.f = scaled;
        output.shape = input.shape;
        output.name = name;
        return output;
    }

    Tensor sum_layer(const Tensor &t1, const Tensor &t2, const std::string &name) {
        assert(t1.shape == t2.shape);
        Func summed;
        summed(c, i, j) = t1.f(c, i, j) + t2.f(c, i, j);
        Tensor output;
        output.f = summed;
        output.shape = t1.shape;
        output.name = name;
        return output;
    }

    Func softmax_layer(const Tensor &input, const int classes, const std::string &name) {
        assert(input.shape[0] == classes);
        RDom r(0, classes);
        Func exp_vals;
        exp_vals(c) = exp(input.f(c));
        Func output("output");
        output(c) = exp_vals(c) / sum(exp_vals(r.x));
        return output;
    }
}