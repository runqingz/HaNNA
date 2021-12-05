// Convolutional Layer on GPU
// (See tf.nn.conv2d)

// On linux, you can compile and run it like so:
// g++ conv2d_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o conv2d_gpu
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./conv2d_gpu [autoscheduler]
//   autoscheduler: String, name of autoscheduler, current supports: Li2018. When not provided, use manual schedule.

// On os x:
// g++ conv2d_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o conv2d_gpu
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./conv2d_gpu [autoscheduler]
//   autoscheduler: String, name of autoscheduler, current supports: Li2018. When not provided, use manual schedule.

#include <stdio.h>
#include <string>
#include "Halide.h"
#include "clock.h"

using namespace std;
using namespace Halide;
using namespace Halide::Tools;

Target find_gpu_target();
void print_4d_buffer(Buffer<float> buf);

// Conv2DLayerGPU follows the tf.nn.conv2d implementation of conv2d layer.
// Note that in this implementation no bias or activation function is applied.
// Data must follows NHWC format.
class Conv2DLayerGPU {
public:
    Var n, x, y, ci, co;
    Func pad, conv;
    Buffer<float> input, filters;
    const int stride;
    const string scheduler;
    RDom r;
    Pipeline autoconv;

    // Constructor parameters:
    //  input: 4-D tensor of shape [batch_size, in_height, in_width, in_channels].
    //  filters: 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels].
    //  stride: Int for the stride of the sliding window.
    //  scheduler: String for the name of the autoscheduler to be used. Empty for manual schedule.
    Conv2DLayerGPU(Buffer<float> input, Buffer<float> filters, const int stride, const string scheduler)
        : input(input), filters(filters), stride(stride), scheduler(scheduler){
       
        // Make sure we have a square kernel.
        assert(filters.dim(0).extent() == filters.dim(1).extent());
        // Make sure we have a valid kernel (size is odd).
        assert(filters.dim(0).extent() % 2 == 1);
        // Make sure the kernel has the same channels as input.
        assert(filters.dim(2).extent() == input.dim(3).extent());

        // Pad input.
        pad = BoundaryConditions::constant_exterior(input, 0);

        // Apply filters.
        const int kernel_size = filters.dim(0).extent();
        const int offset = kernel_size / 2;
        const int in_channels = input.dim(3).extent();
        r = RDom(0, kernel_size, 0, kernel_size, 0, in_channels);
        
        conv(n, x, y, co) += filters(r.x, r.y, r.z, co) * pad(n, stride * x + r.x - offset, stride * y + r.y - offset, r.z);
        
        // Pipline for autoscheduler. Will be skipped if autoscheduler is not used.
        if (! scheduler.empty()){
            autoconv = Pipeline(conv);
        }
    }
        
    // Get a buffer with the shape of the output.
    Buffer<float> get_output_buffer() {
        Buffer<float> output(input.dim(0).extent(), input.dim(1).extent(), input.dim(2).extent(), filters.dim(3).extent());
        return output;
    }

    // Now a schedule that uses CUDA or OpenCL.
    bool schedule_for_gpu() {
        Var xo, yo, xi, yi, tile_index, to, ti, tio, coo, tii, coi;
        
        Target target = find_gpu_target();
        if (!target.has_gpu_feature()) {
            return false;
        }
        
        if (scheduler.empty()) {
            if (target.has_feature(Target::CUDA)) {
                /*conv.fuse(n, x, x)
                    .tile(x, y, xo, yo, xi, yi, 16, 16)
                    .gpu_blocks(xo, yo)
                    .gpu_threads(xi, yi);*/

                /*conv
                    .fuse(n, x, x)
                    .fuse(y, co, y)
                    .tile(x, y, xo, yo, xi, yi, 16, 16)
                    .gpu_blocks(xo, yo)
                    .gpu_threads(xi, yi);*/

                conv
                    .fuse(n, x, x)
                    .tile(x, y, xo, yo, xi, yi, 16, 16)
                    .fuse(xo,yo,tile_index)
                    .gpu_blocks(tile_index, co)
                    .gpu_lanes(xi);

                pad.compute_at(conv, co);
            } else {
                conv.tile(x, y, xo, yo, xi, yi, 32, 32)
                    .fuse(xo, yo, tile_index)
                    .gpu_blocks(tile_index)
                    .gpu_threads(xi);
            }
            

            printf("Target: %s\n", target.to_string().c_str());
            conv.compile_jit(target);
        } else {
            pad.set_estimates({
                {0, input.dim(0).extent()},
                {0, input.dim(1).extent()},
                {0, input.dim(2).extent()},
                {0, input.dim(3).extent()}});

            conv.set_estimates({
                {0, input.dim(0).extent()},
                {0, input.dim(1).extent()},
                {0, input.dim(2).extent()},
                {0, filters.dim(3).extent()}});
            
            autoconv.auto_schedule(scheduler, target);
            autoconv.compile_jit(target);
        }

        return true;
    }

    void test_performance(int num_runs=100) {
        // Test the performance of the scheduled Conv2DLayerGPU.
        Buffer<float> output = this->get_output_buffer();

        // Run the filter once to initialize any GPU runtime state.
        if (scheduler.empty()) {
            conv.realize(output);
        } else {
            autoconv.realize(output);
        }

        // Run pipeline for multiple times.
        double total_time = 0.0;
        double best_time = 0.0;
        for (int i = 0; i < num_runs; i++) {

            double t1 = 0;
            double t2 = 0;

            if (scheduler.empty()) {
                t1 = current_time();
                conv.realize(output);
                output.device_sync();
                t2 = current_time();
            } else {
                t1 = current_time();
                autoconv.realize(output);
                output.device_sync();
                t2 = current_time();
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

    void print_result() {
        // Print result for correctness check.
        Buffer<float> output = this->get_output_buffer();

        // Run pipeline once.
        if (scheduler.empty()) {
            conv.realize(output);
        }
        else {
            autoconv.realize(output);
        }

        output.copy_to_host();

        // Print output to standard out.
        print_4d_buffer(output);
    }
};

int main(int argc, char **argv) {
    // Params:
    //   batch_size: number of images (in a single batch).
    //   channels_in: number of input channels (depth of the input).
    //   channels_out: number of ouput channels (the number of filters).
    //   height: height of the image.
    //   width: width of the image.
    //   kernel_size: width and height of the filters. (3 for 3 x 3 conv layer).
    //   stride: the stride for sliding window.
    const int batch_size = 5, width = 80, height = 100, channels_in = 128, channels_out = 128, kernel_size = 3, stride = 1;
    string scheduler = "";
    bool check = false;
    
    if (argc == 2) {
        string arg = argv[1];
        if (arg == "Li2018") {
            printf("Running performance test for Conv2DLayerGPU with autoscheduler: %s.\n", arg);
            scheduler = arg;
            load_plugin("autoschedule_li2018");
        }
        else if (arg == "--check" || arg == "-c") {
            printf("Running correctness check.\n");
            check = true;
        }
    } else if (argc == 1) {
        printf("Running performance test for Conv2DLayerGPU with manual schedule.\n");
    } else {
        fprintf(stderr, "Usage: .//conv2d_gpu [autoscheduler]\n");
        fprintf(stderr, "       .//conv2d_gpu [--check/-c]\n");
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
                    if (check) {
                        input(b, h, w, c) = 0.1f * (b + h + w + c);
                    }
                    else {
                        input(b, h, w, c) = rand();
                    }
                }
            }
        }
    }

    // Generate random filters.
    printf("Generating filters with dimensions: height: %d, width: %d, channels: %d, num_filters: %d\n", kernel_size, kernel_size, channels_in, channels_out);

    Buffer<float> filters(kernel_size, kernel_size, channels_in, channels_out);
    for (int x = 0; x < kernel_size; x++) {
        for (int y = 0; y < kernel_size; y++) {
            for (int ci = 0; ci < channels_in; ci++) {
                for (int co = 0; co < channels_out; co++) {
                    if (check) {
                        filters(x, y, ci, co) = 0.1f * (x + y + ci + co);
                    }
                    else {
                        filters(x, y, ci, co) = rand();
                    }
                }
            }
        }
    }

    printf("Running pipeline on GPU:\n");
    Conv2DLayerGPU conv_layer(input, filters, stride, scheduler);
    printf("%s", scheduler.c_str());

    conv_layer.schedule_for_gpu();
    if (check) {
        conv_layer.print_result();
    }
    else {
        conv_layer.test_performance(10);
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
    } else {
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