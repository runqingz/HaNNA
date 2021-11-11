// Maxpool Layer on GPU
// (See tf.nn.max_pool())

// On linux, you can compile and run it like so:
// g++ maxpool_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o maxpool_gpu
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./maxpool_gpu.cpp [autoscheduler]
//   autoscheduler: String, name of autoscheduler, current supports: Li2018. When not provided, use manual schedule.

// On os x:
// g++ maxpool_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o maxpool_gpu
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./maxpool_gpu.cpp [autoscheduler]
//   autoscheduler: String, name of autoscheduler, current supports: Li2018. When not provided, use manual schedule.

#include <stdio.h>
#include <string>
#include "Halide.h"
#include "clock.h"

using namespace std;
using namespace Halide;
using namespace Halide::Tools;

Target find_gpu_target();

// MaxpoolLayerGPU follows the tf.nn.max_pool() implementation of maxpool layer.
// Note that in this implementation no bias or activation function is applied.
// Data must follows NHWC format.
class MaxpoolLayerGPU {
public:
    Var n, x, y, ci;
    Func maxpool;
    Buffer<float> input;
    const int stride, kernel_size;
    const string scheduler;
    RDom r;
    Pipeline automaxpool;

    // Constructor parameters:
    //  input: 4-D tensor of shape [batch_size, in_height, in_width, in_channels].
    //  kernel_size: max pooling window size [kernel_size, kernel_size]
    //  stride: Int for the stride of the sliding window.
    //  scheduler: String for the name of the autoscheduler to be used. Empty for manual schedule.
    MaxpoolLayerGPU(Buffer<float> input, int kernel_size, const int stride, const string scheduler)
        : input(input), kernel_size(kernel_size), stride(stride), scheduler(scheduler){

        r = RDom(0, kernel_size, 0, kernel_size);

        //This is same as tf.nn.max_pool with VALID padding, where we contraint halide not try to read outside of image
        maxpool(n, x, y, ci) = input(n, stride * x, stride * y, ci);
        maxpool(n, x, y, ci) = max(input(n, stride*x + r.x, stride*y + r.y, ci), maxpool(n, x, y, ci));
        
        // Pipline for autoscheduler. Will be skipped if autoscheduler is not used.
        if (! scheduler.empty()){
            automaxpool = Pipeline(maxpool);
        }
    }
        
    // Get a buffer with the shape of the output.
    Buffer<float> get_output_buffer() {
        //Boundaries applied
        Buffer<float> output(input.dim(0).extent(), input.dim(1).extent() - kernel_size + 1, input.dim(2).extent() - kernel_size + 1, input.dim(3).extent());
        output.set_min(0,0,0,0);
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
                maxpool.tile(x, y, x_outer, y_outer, x_inner, y_inner, 32, 32)
                .fuse(x_outer, y_outer, tile_index)
                .gpu_blocks(tile_index)
                .gpu_threads(x_inner);
            } else {
                maxpool.tile(x, y, x_outer, y_outer, x_inner, y_inner, 32, 32)
                .fuse(x_outer, y_outer, tile_index)
                .gpu_blocks(tile_index)
                .gpu_threads(x_inner);
            }
            

            printf("Target: %s\n", target.to_string().c_str());
            maxpool.compile_jit(target);
        } else {
            maxpool.set_estimates({
                {0, input.dim(0).extent()},
                {0, input.dim(1).extent()},
                {0, input.dim(2).extent()},
                {0, input.dim(3).extent()}});
            
            automaxpool.auto_schedule(scheduler, target);
            automaxpool.compile_jit(target);
        }

        return true;
    }

    void test_performance(int num_runs=100) {
        // Test the performance of the scheduled MaxpoolLayerGPU.
        Buffer<float> output = this->get_output_buffer();

        // Run the filter once to initialize any GPU runtime state.
        if (scheduler.empty()) {
            maxpool.realize(output);
        } else {
            automaxpool.realize(output);
        }

        // Run pipeline for multiple times.
        double total_time = 0.0;
        double best_time = 0.0;
        for (int i = 0; i < num_runs; i++) {

            double t1 = current_time();

            if (scheduler.empty()) {
                maxpool.realize(output);
            } else {
                automaxpool.realize(output);
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
    //   height: height of the image.
    //   width: width of the image.
    //   kernel_size: width and height of the filters. (3 for 3 x 3 max pool layer).
    //   stride: the stride for sliding window.
    const int batch_size = 8, width = 120, height = 100, channels_in = 3, channels_out = 3, kernel_size = 3, stride = 1;
    string scheduler = "";
    
    if (argc == 2) {
        printf("Running performance test for MaxpoolLayerGPU with autoscheduler: %s.\n", argv[1]);
        scheduler = argv[1];
        load_plugin("autoschedule_li2018");
    } else if (argc == 1) {
        printf("Running performance test for MaxpoolLayerGPU with manual schedule.\n");
    } else {
        fprintf(stderr, "Usage: .//maxpool_gpu [autoscheduler]\n");
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

    printf("Running pipeline on GPU:\n");
    MaxpoolLayerGPU maxpool_layer(input, kernel_size, stride, scheduler);
    printf("%s", scheduler.c_str());

    maxpool_layer.schedule_for_gpu();
    maxpool_layer.test_performance();
    
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
