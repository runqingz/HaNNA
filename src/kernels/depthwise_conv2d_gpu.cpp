// Depthwise Convolutional Layer on GPU
// (See tf.nn.depthwise_conv2d)

// On linux, you can compile and run it like so:
// g++ depthwise_conv2d_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o depthwise_conv2d_gpu
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./depthwise_conv2d_gpu <autoscheduler>
// autoscheuler: String, name of autoscheduler, current supports: Li2018

// On os x:
// g++ depthwise_conv2d_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o depthwise_conv2d_gpu
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./depthwise_conv2d_gpu <autoscheduler>
// autoscheuler: String, name of autoscheduler, current supports: Li2018
#include <stdio.h>
#include <string>
#include "Halide.h"
#include "clock.h"

using namespace std;
using namespace Halide;
using namespace Halide::Tools;

Target find_gpu_target();

// DepthwiseConv2DLayerGPU follows the tf.nn.depthwise_conv2d implementation of
// depthwise conv2d layer.
// Note that in this implementation no bias or activation function is applied.
// Data must follows NHWC format.
class DepthwiseConv2DLayerGPU {
public:
    Var n, x, y, ci, co;
    Func pad, depthwise_conv;
    Buffer<float> input;
    Buffer<float> depthwise_filters;
    const int stride;
    RDom r;
    Pipeline auto_depthwiseconv;

    string scheduler;

    // Constructor parameters:
    //  input: 4-D tensor of shape [batch_size, in_height, in_width, in_channels].
    //  depthwise_filters: 4-D tensor of shape [filter_height, filter_width, in_channels, channel_multiplier].
    //  stride: int for the stride of the sliding window.
    DepthwiseConv2DLayerGPU(Buffer<float> input, Buffer<float> depthwise_filters, const int stride)
        : input(input), depthwise_filters(depthwise_filters), stride(stride), scheduler(scheduler) {

        // Make sure we have a square kernel.
        assert(depthwise_filters.dim(0).extent() == depthwise_filters.dim(1).extent());
        // Make sure we have a valid kernel (size is odd).
        assert(depthwise_filters.dim(1).extent() % 2 == 1);
        // Make sure the kernel has the same channels as input.
        assert(depthwise_filters.dim(2).extent() == input.dim(3).extent());

        // Pad input.
        pad = BoundaryConditions::constant_exterior(input, 0);

        // Apply filters.
        const int kernel_size = depthwise_filters.dim(1).extent();
        const int offset = kernel_size / 2;
        const int channel_multiplier = depthwise_filters.dim(3).extent();
        r = RDom(0, kernel_size, 0, kernel_size);
        
        depthwise_conv(n, x, y, co) += depthwise_filters(r.x, r.y, co / channel_multiplier, co % channel_multiplier) * pad(n, stride * x + r.x - offset, stride * y + r.y - offset, co / channel_multiplier);
        auto_depthwiseconv = Pipeline(depthwise_conv);
    }
    
    // Get a buffer with the shape of the output.
    Buffer<float> get_output_buffer() {
        Buffer<float> output(input.dim(0).extent(), input.dim(1).extent(), input.dim(2).extent(), depthwise_filters.dim(2).extent() * depthwise_filters.dim(3).extent());
        return output;
    }

    // Now a schedule that uses CUDA or OpenCL.
    bool schedule_for_gpu() {
        Var x_outer, y_outer, x_inner, y_inner, tile_index;
        
        Target target = find_gpu_target();
        if (!target.has_gpu_feature()) {
            return false;
        }

        if (scheduler.empty()) {
            if (target.has_feature(Target::CUDA)) {
                //CUDA will use cuda specific derivatives such as gpu_lane
                depthwise_conv.tile(x, y, x_outer, y_outer, x_inner, y_inner, 32, 32)
                    .fuse(x_outer, y_outer, tile_index)
                    .gpu_blocks(tile_index)
                    .gpu_threads(x_inner);
            }
            else {
                depthwise_conv.tile(x, y, x_outer, y_outer, x_inner, y_inner, 32, 32)
                    .fuse(x_outer, y_outer, tile_index)
                    .gpu_blocks(tile_index)
                    .gpu_threads(x_inner);
            }

            printf("Target: %s\n", target.to_string().c_str());
            depthwise_conv.compile_jit(target);
        }
        else {
            pad.set_estimates({
                {0, input.dim(0).extent()},
                {0, input.dim(1).extent()},
                {0, input.dim(2).extent()},
                {0, input.dim(3).extent()} });

            depthwise_conv.set_estimates({
                {0, input.dim(0).extent()},
                {0, input.dim(1).extent()},
                {0, input.dim(2).extent()},
                {0, depthwise_filters.dim(2).extent() * depthwise_filters.dim(3).extent()} });

            auto_depthwiseconv.auto_schedule(scheduler, target);
            auto_depthwiseconv.compile_jit(target);
        }

        return true;
    }

    void test_performance(int num_runs=100) {
        // Test the performance of the scheduled DepthwiseConv2DLayerGPU.
        Buffer<float> output = this->get_output_buffer();

        // Run the filter once to initialize any GPU runtime state.
        if (scheduler.empty()) {
            depthwise_conv.realize(output);
        } else {
            auto_depthwiseconv.realize(output);
        }

        // Run pipeline for multiple times.
        double total_time = 0.0;
        double best_time = 0.0;
        for (int i = 0; i < num_runs; i++) {

            double t1 = current_time();
            if (scheduler.empty()) {
                depthwise_conv.realize(output);
            } else {
                auto_depthwiseconv.realize(output);
            }

            // Force any GPU code to finish by copying the buffer back to the CPU.
            output.device_sync();

            double t2 = current_time();

            double elapsed = (t2 - t1);
            if (i == 0 || elapsed < best_time) {
                best_time = elapsed;
            }
            total_time += elapsed;
        }
        printf("%d runs in total\n", num_runs);
        printf("Average: %1.4f milliseconds\n", total_time / num_runs);
        printf("Best: %1.4f milliseconds\n", best_time);
    }
};

int main(int argc, char **argv) {
    // Params:
    //   batch_size: number of images (in a single batch).
    //   channels_in: number of input channels (depth of the input).
    //   channel_multiplier: number of filters applied to each input channel (output has channels_in * channel_multiplier channels).
    //   height: height of the image.
    //   width: width of the image.
    //   kernel_size: width and height of the filters. (3 for 3 x 3 conv layer).
    //   stride: the stride for sliding window.
    const int batch_size = 8, width = 120, height = 100, channels_in = 3, channel_multiplier = 2, kernel_size = 5, stride = 1;
    bool auto_schedule = true;
    string scheduler = "";

    if (argc == 2) {
        printf("Running performance test for DepthwiseConv2DLayerGPU with autoscheduler: %s.\n", argv[1]);
        scheduler = argv[1];
        load_plugin("autoschedule_li2018");
    }
    else if (argc == 1) {
        printf("Running performance test for DepthwiseConv2DLayerGPU with manual schedule.\n");
    }
    else {
        fprintf(stderr, "Usage: .//depthwise_conv2d_gpu [autoscheduler]\n");
        return 1;
    }

    // Generate random input.
    // Input shape follows TensorFlow convention (N, H, W, C)
    printf("Generating input with dimensions: batch_size: %d, height: %d, width: %d, channels: %d\n", batch_size, height, width, channels_in);

    Buffer<float> input(batch_size, height, width, channels_in);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels_in; c++) {
                    input(b, h, w, c) = rand();
                }
            }
        }
    }

    // Generate random filters.
    printf("Generating filters with dimensions: height: %d, width: %d, channels: %d, channels multiplier: %d\n", kernel_size, kernel_size, channels_in, channel_multiplier);

    Buffer<float> depthwise_filters(kernel_size, kernel_size, channels_in, channel_multiplier);
    for (int x = 0; x < kernel_size; x++) {
        for (int y = 0; y < kernel_size; y++) {
            for (int ci = 0; ci < channels_in; ci++) {
                for (int cm = 0; cm < channel_multiplier; cm++) {
                    depthwise_filters(x, y, ci, cm) = rand();
                }         
            }
        }
    }

    printf("Running pipeline on GPU:\n");
    DepthwiseConv2DLayerGPU depthwise_conv_layer(input, depthwise_filters, stride);

    depthwise_conv_layer.schedule_for_gpu();
    printf("Testing performance on GPU:\n");
    depthwise_conv_layer.test_performance();

    return 0;
}

// A helper function to check if OpenCL, Metal or D3D12 is present on the host machine.

Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Target target = get_host_target();

    vector<Target::Feature> features_to_try;
    
    if (target.os == Target::OSX) {
        // OS X doesn't update its OpenCL drivers, so they tend to be broken.
        // CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.push_back(Target::Metal);
    } else {
        features_to_try.push_back(Target::CUDA);
    }
    // Uncomment the following lines to also try CUDA:
    // features_to_try.push_back(Target::CUDA);

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }

    printf("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)\n");
    return target;
}
