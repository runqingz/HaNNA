// Relu Layer on GPU
// (See tf.nn.relu)

// On linux, you can compile and run it like so:
// g++ relu_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o relu_gpu
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./relu_gpu <autoscheduler>
// autoscheuler: String, name of autoscheduler, current supports: Li2018

// On os x:
// g++ relu_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o relu_gpu
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./relu_gpu  <autoscheduler>
// autoscheuler: String, name of autoscheduler, current supports: Li2018
#include <stdio.h>
#include <string>
#include "Halide.h"
#include "clock.h"

using namespace std;
using namespace Halide;
using namespace Halide::Tools;

Target find_gpu_target();
void print_4d_buffer(Buffer<float> buf);

// Relu follows the tf.nn.relu implementation of
// relu activation layer.
// Data must follows NHWC format.
class ReluLayerGPU {
public:
    Var n, x, y, ci;
    Func relu;
    Buffer<float> input;
    Pipeline auto_relu;

    const string scheduler;

    // Constructor parameters:
    //  input: 4-D tensor of shape [batch_size, in_height, in_width, in_channels].
    ReluLayerGPU(Buffer<float> input, string scheduler)
        : input(input), scheduler(scheduler) {

        relu(n, ci, x, y) = max(input(n, ci, x, y) , 0.f);
        auto_relu = Pipeline(relu);
    }

    // Get a buffer with the shape of the output.
    Buffer<float> get_output_buffer() {
        Buffer<float> output(input.dim(0).extent(), input.dim(1).extent(), input.dim(2).extent(), input.dim(3).extent());
        return output;
    }

    // Now a schedule that uses CUDA or OpenCL.
    bool schedule_for_gpu() {
        Var xo, yo, xi, yi, tile_index, nc, nco, nci;

        Target target = find_gpu_target();
        if (!target.has_gpu_feature()) {
            return false;
        }

        if (scheduler.empty()) {
            if (target.has_feature(Target::CUDA)) {
                /*relu.tile(x, y, xo, yo, xi, y_inner, 32, 32)
                    .fuse(xo, yo, tile_index)
                    .gpu_blocks(tile_index)
                    .gpu_threads(xi);*/

                /*relu.fuse(n, ci, nc)
                    .tile(nc, x, nco, xo, nci, xi, 16, 16)
                    .gpu_blocks(nco, xo)
                    .gpu_threads(nci);*/

                //seems to be the best so far
                relu.fuse(n, ci, nc)
                    .tile(nc, x, nco, xo, nci, xi, 32, 32)
                    .gpu_blocks(nco, y)
                    .gpu_threads(nci, xi);

                /*relu.fuse(n, ci, nc)
                    .tile(x, y, xo, yo, xi, yi, 8, 8)
                    .gpu_blocks(nc, xo, yo)
                    .gpu_threads(xi, yi);*/
            }
            else {
                /*relu.tile(x, y, x_outer, y_outer, x_inner, y_inner, 32, 32)
                    .fuse(x_outer, y_outer, tile_index)
                    .gpu_blocks(tile_index)
                    .gpu_threads(x_inner);*/
            }

            printf("Target: %s\n", target.to_string().c_str());
            relu.compile_jit(target);
        }
        else {
            relu.set_estimates({
               {0, input.dim(0).extent()},
               {0, input.dim(1).extent()},
               {0, input.dim(2).extent()},
               {0, input.dim(3).extent()} });

            auto_relu.auto_schedule(scheduler, target);
            auto_relu.compile_jit(target);
        }
        
        return true;
    }

    void print_result() {
        // Print result for correctness check.
        Buffer<float> output = this->get_output_buffer();

        // Run pipeline once.
        if (scheduler.empty()) {
            relu.realize(output);
            output.copy_to_host();
        }
        else {
            auto_relu.realize(output);
        }

        // Print output to standard out.
        print_4d_buffer(output);
    }

    void test_performance(int num_runs = 100) {
        // Test the performance of the scheduled ReluLayerGPU.
        printf("Testing performance on GPU:\n");
        Buffer<float> output = this->get_output_buffer();

        // Run the filter once to initialize any GPU runtime state.
        if (scheduler.empty()) {
            relu.realize(output);
        }
        else {
            auto_relu.realize(output);
        }

        // Run pipeline for multiple times.
        double total_time = 0.0;
        double best_time = 0.0;
        for (int i = 0; i < num_runs; i++) {
            double t1 = 0;
            double t2 = 0;
            
            if (scheduler.empty()) {
                t1 = current_time();
                relu.realize(output);
                output.device_sync();
                t2 = current_time();
            }
            else {
                auto_relu.realize(output);
            }

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

int main(int argc, char** argv) {
    // Params:
    //   batch_size: number of images (in a single batch).
    //   channels_in: number of input channels (depth of the input).
    //   height: height of the image.
    //   width: width of the image.
    string scheduler = "";
    bool check = false;

    if (argc == 2) {
        string arg = argv[1];
        if (arg == "") {
            printf("Running performance test for ReluLayerGPU with autoscheduler: %s.\n", argv[1]);
            scheduler = arg;
            load_plugin("autoschedule_li2018");
        }
        else if (arg == "--check" || arg == "-c") {
            printf("Running correctness check.\n");
            check = true;
        }
    }
    else if (argc == 1) {
        printf("Running performance test for ReluLayerGPU with manual schedule.\n");
    }
    else {
        fprintf(stderr, "Usage: .//relu_gpu [autoscheduler]\n");
        return 1;
    }

    const int batch_size = check ? 1 : 8, width = check ? 8: 256, height = check ? 8 : 256, channels_in = check ? 2 : 128;

    // Generate random input.
    // Input shape follows TensorFlow convention (N, H, W, C)
    printf("Generating input with dimensions: batch_size: %d, height: %d, width: %d, channels: %d\n", batch_size, height, width, channels_in);

    Buffer<float> input(batch_size, channels_in, height, width);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels_in; c++) {
                    if (check) {
                        input(b, c, h, w) = rand();
                    }
                    else {
                        input(b, h, w, c) = 0.1f * (b + h + w + c);
                    }
                }
            }
        }
    }

    printf("Running pipeline on GPU:\n");
    ReluLayerGPU relu_layer(input, scheduler);

    relu_layer.schedule_for_gpu();
    if (check) {
        relu_layer.print_result();
    }
    else {
        relu_layer.test_performance();
    }

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
    }
    else {
        features_to_try.push_back(Target::CUDA);
    }

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }

    printf("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)\n");
    return target;
}

void print_4d_buffer(Buffer<float> buf) {
    // Print out the values in realized buffer.
    // Input must be a 4-dimensional array of floats.
    const int b_lo = buf.min(0);
    const int b_hi = b_lo + buf.extent(0);
    const int h_lo = buf.min(1);
    const int h_hi = h_lo + buf.extent(1);
    const int w_lo = buf.min(2);
    const int w_hi = w_lo + buf.extent(2);
    const int c_lo = buf.min(3);
    const int c_hi = c_lo + buf.extent(3);

    std::cout << "[";
    for (int b = b_lo; b < b_hi; b++) {
        std::cout << "[";
        for (int h = h_lo; h < h_hi; h++) {
            std::cout << "[";
            for (int w = w_lo; w < w_hi; w++) {
                std::cout << "[";
                for (int c = c_lo; c < c_hi; c++) {
                    printf("%.2f ", buf(b, h, w, c));
                }
                std::cout << "]\n";
            }
            std::cout << "]";
        }
        std::cout << "]";
    }
    std::cout << "]\n";
}
