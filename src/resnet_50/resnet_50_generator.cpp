#include "Halide.h"
#include <tuple>
#include <unordered_map>
#include "kernel.hpp"

using namespace Kernel;
namespace {

// returns index of found value in array or -1 if not in array
int find_index(int value, std::vector<int> vec) {
    std::vector<int>::iterator it = std::find(vec.begin(), vec.end(), value);
    if (it == vec.end())
        return -1;
    return std::distance(vec.begin(), it);
}

class Resnet50Generator : public Halide::Generator<Resnet50Generator> {
public:
    Input<Buffer<float>> input{"input", 3};
    /** parameter values for scaling layers **/
    Input<Buffer<float>> conv1_gamma{"conv1_gamma", 1};
    Input<Buffer<float>[4]> br1_gamma { "br1_gamma", 1 };
    Input<Buffer<float>[16]> br2a_gamma { "br2a_gamma", 1 };
    Input<Buffer<float>[16]> br2b_gamma { "br2b_gamma", 1 };
    Input<Buffer<float>[16]> br2c_gamma { "br2c_gamma", 1 };

    Input<Buffer<float>> conv1_beta{"conv1_beta", 1};
    Input<Buffer<float>[4]> br1_beta { "br1_beta", 1 };
    Input<Buffer<float>[16]> br2a_beta { "br2a_beta", 1 };
    Input<Buffer<float>[16]> br2b_beta { "br2b_beta", 1 };
    Input<Buffer<float>[16]> br2c_beta { "br2c_beta", 1 };

    Input<Buffer<float>> conv1_mu{"conv1_mu", 1};
    Input<Buffer<float>[4]> br1_mu { "br1_mu", 1 };
    Input<Buffer<float>[16]> br2a_mu { "br2a_mu", 1 };
    Input<Buffer<float>[16]> br2b_mu { "br2b_mu", 1 };
    Input<Buffer<float>[16]> br2c_mu { "br2c_mu", 1 };

    Input<Buffer<float>> conv1_sig{"conv1_sig", 1};
    Input<Buffer<float>[4]> br1_sig { "br1_sig", 1 };
    Input<Buffer<float>[16]> br2a_sig { "br2a_sig", 1 };
    Input<Buffer<float>[16]> br2b_sig { "br2b_sig", 1 };
    Input<Buffer<float>[16]> br2c_sig { "br2c_sig", 1 };

    /** weights and biases for convolutions **/
    Input<Buffer<float>> conv1_weights{"conv1_weights", 4};
    Input<Buffer<float>[4]> br1_conv_weights { "br1_conv_weights", 4 };
    Input<Buffer<float>[16]> br2a_conv_weights { "br2a_conv_weights", 4 };
    Input<Buffer<float>[16]> br2b_conv_weights { "br2b_conv_weights", 4 };
    Input<Buffer<float>[16]> br2c_conv_weights { "br2c_conv_weights", 4 };

    Input<Buffer<float>> fc1000_weights{"fc1000_weights", 2};
    Input<Buffer<float>> fc1000_bias{"fc1000_bias", 1};
    Output<Buffer<float>> final_output{"final_output", 1};

    /** list out shapes of each layers weights **/
    // weight shapes: out channels, kernel_w, kernel_h, pad, stride. In channels infered by input tensor shape
    const WeightShape conv1_ws = {64, 7, 7, 3, 2};
    const WeightShape pool1_ws = {64, 3, 3, 1, 2};
    const WeightShape pool5_ws = {2048, 7, 7, 0, 1};
    const WeightShape fc1000_ws = {1000, 1, 1, 0, 1};  // 1x1 conv with 2048 input channels and 1000 output channels

    // res2a, res2b, res2c all have shame shapes
    const WeightShape res2x_br2a_ws = {64, 1, 1, 0, 1};
    const WeightShape res2a_br2b_ws = {64, 3, 3, 1, 1};
    const WeightShape res2x_br2b_ws = {64, 3, 3, 1, 1};
    const WeightShape res2x_br2c_ws = {256, 1, 1, 0, 1};
    const WeightShape res2a_br1_ws = {256, 1, 1, 0, 1};

    // res3x is same for most layers
    const WeightShape res3x_br2a_ws = {128, 1, 1, 0, 1};
    const WeightShape res3a_br2b_ws = {128, 3, 3, 1, 2};
    const WeightShape res3x_br2b_ws = {128, 3, 3, 1, 1};
    const WeightShape res3x_br2c_ws = {512, 1, 1, 0, 1};
    const WeightShape res3a_br1_ws = {512, 1, 1, 0, 2};

    const WeightShape res4x_br2a_ws = {256, 1, 1, 0, 1};
    const WeightShape res4a_br2b_ws = {256, 3, 3, 1, 2};
    const WeightShape res4x_br2b_ws = {256, 3, 3, 1, 1};
    const WeightShape res4x_br2c_ws = {1024, 1, 1, 0, 1};
    const WeightShape res4a_br1_ws = {1024, 1, 1, 0, 2};

    const WeightShape res5x_br2a_ws = {512, 1, 1, 0, 1};
    const WeightShape res5a_br2b_ws = {512, 3, 3, 1, 2};
    const WeightShape res5x_br2b_ws = {512, 3, 3, 1, 1};
    const WeightShape res5x_br2c_ws = {2048, 1, 1, 0, 1};
    const WeightShape res5a_br1_ws = {2048, 1, 1, 0, 2};

    const WeightShape br1_ws[4] = {res2a_br1_ws, res3a_br1_ws, res4a_br1_ws, res5a_br1_ws};
    const WeightShape br2a_ws[16] = {res2x_br2a_ws, res2x_br2a_ws, res2x_br2a_ws,
                                     res3x_br2a_ws, res3x_br2a_ws, res3x_br2a_ws, res3x_br2a_ws,
                                     res4x_br2a_ws, res4x_br2a_ws, res4x_br2a_ws, res4x_br2a_ws, res4x_br2a_ws, res4x_br2a_ws,
                                     res5x_br2a_ws, res5x_br2a_ws, res5x_br2a_ws};
    const WeightShape br2b_ws[16] = {res2a_br2b_ws, res2x_br2b_ws, res2x_br2b_ws,
                                     res3a_br2b_ws, res3x_br2b_ws, res3x_br2b_ws, res3x_br2b_ws,
                                     res4a_br2b_ws, res4x_br2b_ws, res4x_br2b_ws, res4x_br2b_ws, res4x_br2b_ws, res4x_br2b_ws,
                                     res5a_br2b_ws, res5x_br2b_ws, res5x_br2b_ws};
    const WeightShape br2c_ws[16] = {res2x_br2c_ws, res2x_br2c_ws, res2x_br2c_ws,
                                     res3x_br2c_ws, res3x_br2c_ws, res3x_br2c_ws, res3x_br2c_ws,
                                     res4x_br2c_ws, res4x_br2c_ws, res4x_br2c_ws, res4x_br2c_ws, res4x_br2c_ws, res4x_br2c_ws,
                                     res5x_br2c_ws, res5x_br2c_ws, res5x_br2c_ws};

    Var c, i, j;

    void generate() {

        // Algorithm

        /** Declare arrays of other functions and build the requested block **/
        Tensor br1_conv[4];
        Tensor br1_norm[4];
        Tensor br1_scale[4];

        Tensor br2a_conv[16];
        Tensor br2a_norm[16];
        Tensor br2a_scaled[16];
        Tensor br2a_relu[16];

        Tensor br2b_conv[16];
        Tensor br2b_norm[16];
        Tensor br2b_scaled[16];
        Tensor br2b_relu[16];

        Tensor br2c_conv[16];
        Tensor br2c_norm[16];
        Tensor br2c_scaled[16];

        Tensor resunit_sum[16];
        Tensor resunit_relu[16];

        Tensor pool5;
        Tensor fc1000;
        Tensor softmax;

        // these tensors are different depending on the block and must be conditionally assigned.
        Tensor input_t;
        std::vector<int> input_shape;
        Tensor br2a_input;
        Tensor resunit_sum_input;

        // used only for block_id == 0
        Tensor conv1, norm1, scaled1, relu1, pool1;

        std::vector<int> branch1_indices{0, 3, 7, 13};

        /** if block_id is 0 build the (stem) conv1 section **/
        for (int block_id = 0; block_id < 16; ++block_id) {
            if (block_id == 0) {
                input_shape = {3, 224, 224};
                input_t.f = input;
                input_t.shape = input_shape;

                conv1 = conv2D(input_t, conv1_ws, conv1_weights, "conv1");
                norm1 = norm_layer(conv1, conv1_mu, conv1_sig, "norm1");
                scaled1 = scale_layer(norm1, conv1_gamma, conv1_beta, "scale1");
                relu1 = relu_layer(scaled1, "relu1");
                pool1 = max_pool_layer(relu1, pool1_ws, "pool1");

                br2a_input = pool1;
            } else {
                br2a_input = resunit_relu[block_id - 1];
            }

            // build branch1 if this section has branch1
            int br1_i = find_index(block_id, branch1_indices);
            if (br1_i >= 0) {
                br1_conv[br1_i] = conv2D(br2a_input, br1_ws[br1_i], br1_conv_weights[br1_i], "br1_conv");
                br1_norm[br1_i] = norm_layer(br1_conv[br1_i], br1_mu[br1_i], br1_sig[br1_i], "br1_norm");
                br1_scale[br1_i] = scale_layer(br1_norm[br1_i], br1_gamma[br1_i], br1_beta[br1_i], "br1_scale");
                resunit_sum_input = br1_scale[br1_i];
            } else {
                resunit_sum_input = resunit_relu[block_id - 1];
            }

            // branch2a
            auto weights = br2a_conv_weights[block_id];

            br2a_conv[block_id] = conv2D(br2a_input, br2a_ws[block_id], weights, "block" + std::to_string(block_id) + "_2a_conv");
            br2a_norm[block_id] = norm_layer(br2a_conv[block_id], br2a_mu[block_id], br2a_sig[block_id], "block" + std::to_string(block_id) + "_2a_norm");
            br2a_scaled[block_id] = scale_layer(br2a_norm[block_id], br2a_gamma[block_id], br2a_beta[block_id], "block" + std::to_string(block_id) + "_2a_scale");
            br2a_relu[block_id] = relu_layer(br2a_scaled[block_id], "2a_relu");

            // branch 2b
            weights = br2b_conv_weights[block_id];
            br2b_conv[block_id] = conv2D(br2a_relu[block_id], br2b_ws[block_id], weights, "block" + std::to_string(block_id) + "_2b_conv");
            br2b_norm[block_id] = norm_layer(br2b_conv[block_id], br2b_mu[block_id], br2b_sig[block_id], "block" + std::to_string(block_id) + "_2b_norm");
            br2b_scaled[block_id] = scale_layer(br2b_norm[block_id], br2b_gamma[block_id], br2b_beta[block_id], "block" + std::to_string(block_id) + "_2b_scale");
            br2b_relu[block_id] = relu_layer(br2b_scaled[block_id], "2b_relu");

            // branch 2c
            weights = br2c_conv_weights[block_id];
            br2c_conv[block_id] = conv2D(br2b_relu[block_id], br2c_ws[block_id], weights, "block" + std::to_string(block_id) + "_2c_conv");
            br2c_norm[block_id] = norm_layer(br2c_conv[block_id], br2c_mu[block_id], br2c_sig[block_id], "block" + std::to_string(block_id) + "_2c_norm");
            br2c_scaled[block_id] = scale_layer(br2c_norm[block_id], br2c_gamma[block_id], br2c_beta[block_id], "block" + std::to_string(block_id) + "_2c_scale");

            // create residual unit
            resunit_sum[block_id] = sum_layer(resunit_sum_input, br2c_scaled[block_id], "block" + std::to_string(block_id) + "_res_sum");
            resunit_relu[block_id] = relu_layer(resunit_sum[block_id], "block" + std::to_string(block_id) + "_res_relu");

            // create final 3 layers
            if (block_id == 15) {
                pool5 = avg_pool_layer(resunit_relu[block_id], pool5_ws, "pool5");
                fc1000 = fc_layer(pool5, fc1000_ws, fc1000_weights, fc1000_bias, "fc");
                final_output = softmax_layer(fc1000, 1000, "softmax");
            }
        }

        // TODO: Actually schedule this.
        conv1.f.compute_root();
        scaled1.f.compute_root();
        relu1.f.compute_root();
        pool1.f.compute_root();
        for (int i = 0; i < 16; i++) {
            br2a_relu[i].f.compute_root().vectorize(c, 8).parallel(j);
            br2b_relu[i].f.compute_root().vectorize(c, 8).parallel(j);
            resunit_relu[i].f.compute_root().vectorize(c, 8).parallel(j);
        }
        pool5.f.compute_root();
        fc1000.f.compute_root();
        softmax.f.compute_root();
    }
};
}  //namespace

HALIDE_REGISTER_GENERATOR(Resnet50Generator, resnet50)
