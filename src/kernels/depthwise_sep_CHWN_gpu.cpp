// Depthwise Convolutional Layer on GPU with CHWN data layout
// (See tf.nn.depthwise_conv2d)

// On linux, you can compile and run it like so:
// g++ depthwise_sep_CHWN_conv2d_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o depthwise_sep_CHWN_conv2d_gpu
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./depthwise_sep_CHWN_conv2d_gpu <autoscheduler>
// autoscheuler: String, name of autoscheduler, current supports: Li2018

// On os x:
// g++ depthwise_sep_CHWN_conv2d_gpu.cpp -g -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o depthwise_sep_CHWN_conv2d_gpu
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./depthwise_sep_CHWN_conv2d_gpu <autoscheduler>
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

class DepthwiseSepConv2DLayerCHWNGPU {
public:
    Var n, x, y, ci, co, q;
    Func depthwise_conv, pad, relu, inputf, filtersf;
    Buffer<float> input, depthwise_filters;
    const int stride;
    const string scheduler;
    RDom r;
    Pipeline autoconv;

    // Constructor parameters:
    //  input: 4-D tensor of shape [batch_size, in_height, in_width, in_channels].
    //  filters: 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels].
    //  stride: Int for the stride of the sliding window.
    //  scheduler: String for the name of the autoscheduler to be used. Empty for manual schedule.
    DepthwiseSepConv2DLayerCHWNGPU(Buffer<float> input, Buffer<float> filters, const int stride, const string scheduler)
        : input(input), depthwise_filters(filters), stride(stride), scheduler(scheduler) {
        // Make sure we have a square kernel.
        assert(depthwise_filters.dim(1).extent() == filters.dim(2).extent());
        // Make sure we have a valid kernel (size is odd).
        assert(depthwise_filters.dim(1).extent() % 2 == 1);
        // Make sure the kernel has the same channels as input.
        assert(depthwise_filters.dim(3).extent() == input.dim(0).extent());

        // Pad input.
        pad = BoundaryConditions::constant_exterior(input, 0);

        // Load filters into shared memory.
        //filtersf(co, x, y, ci) = filters(co, x, y, ci);

        // Apply filters.
        const int kernel_size = depthwise_filters.dim(3).extent();
        const int offset = kernel_size / 2;
        const int channel_multiplier = depthwise_filters.dim(0).extent();
        r = RDom(0, kernel_size, 0, kernel_size);

        //BEFORE data layout change
        //depthwise_conv(n, x, y, co) += depthwise_filters(r.x, r.y, co / channel_multiplier, co % channel_multiplier) * pad(n, stride * x + r.x - offset, stride * y + r.y - offset, co / channel_multiplier);
        //depthwise_conv(ci * channel_multiplier + q, x, y, n) += depthwise_filters(r.x, r.y, ci, q) * pad(ci, x + r.x , y + r.y, n);
        depthwise_conv(co, x, y, n) += depthwise_filters(co % channel_multiplier, co / channel_multiplier, r.x, r.y) * pad(co / channel_multiplier, x + r.x , y + r.y, n);
        //relu(co, x, y, n) = max(0, depthwise_conv(co, x, y, n));

        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"), tile_index("tilei"), to("to"), ti("ti"), tio("tio"), coo("coo"), tii("tii"), coi("coi"), s("s"), t("t");
        RVar rxo("rxo"), rxi("rxi"), rxii("rxii"), temp;

        /*relu.compute_root()
            .split(x, xo, xi, 4)
            .split(y, yo, yi, 4)
            .split(co, coo, coi, 32)
            .reorder(xi, yi, coi, xo, yo, coo, n)
            .unroll(xi)
            .unroll(yi)
            .fuse(coo, n, tile_index)
            .gpu_blocks(xo, yo, tile_index)
            .gpu_threads(coi);*/

        /*depthwise_conv.compute_at(relu, xo)
            .update()
            .fuse(y, n, y)
            .split(r.x, rxo, rxi, 32)
            .split(rxi, rxi, rxii, 2)
            .reorder(co, rxii, rxi, rxo, r.y, r.z, x, y)
            .gpu_blocks(x, y)
            .gpu_threads(co)
            .unroll(r.y)
            .unroll(r.z);*/

        /*pad.compute_at(depthwise_conv, rxo)
            .fuse(_0, _1, s)
            .fuse(_2, _3, t)
            .gpu_threads(s);*/

        depthwise_conv.compute_root();

        Target target = find_gpu_target();
        depthwise_conv.compile_jit(target);

        // Pipline for autoscheduler. Will be skipped if autoscheduler is not used.
        if (!scheduler.empty()) {
            throw "Unimplemented";
            //autoconv = Pipeline(conv);
        }
    }

    // Get a buffer with the shape of the output.
    Buffer<float> get_output_buffer() {
        Buffer<float> output(depthwise_filters.dim(0).extent() * depthwise_filters.dim(1).extent(), input.dim(1).extent(), input.dim(2).extent(), input.dim(3).extent());
        return output;
    }

    void test_performance(int num_runs = 100) {
        // Test the performance of the scheduled Conv2DLayerGPU.
        Buffer<float> output = this->get_output_buffer();

        // Run the filter once to initialize any GPU runtime state.
        if (scheduler.empty()) {
            depthwise_conv.realize(output);
        }
        else {
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
                depthwise_conv.realize(output);
                output.device_sync();
                t2 = current_time();
            }
            else {
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
            depthwise_conv.realize(output);
        }
        else {
            autoconv.realize(output);
        }

        output.copy_to_host();

        // Print output to standard out.
        print_4d_buffer(output);
    }
};

int main(int argc, char** argv) {
    // Params:
    //   batch_size: number of images (in a single batch).
    //   channels_in: number of input channels (depth of the input).
    //   channel_multiplier: number of filters applied to each input channel (output has channels_in * channel_multiplier channels).
    //   height: height of the image.
    //   width: width of the image.
    //   kernel_size: width and height of the filters. (3 for 3 x 3 conv layer).
    //   stride: the stride for sliding window.
    
    string scheduler = "";
    bool check = false;

    if (argc == 2) {
        string arg = argv[1];
        if (arg == "Li2018") {
            //Load Auto Scheduler plugins
            printf("Running performance test for Conv2DLayerGPU with autoscheduler: %s.\n", arg);
            scheduler = arg;
            load_plugin("autoschedule_li2018");
        }
        else if (arg == "--check" || arg == "-c") {
            printf("Running correctness check.\n");
            check = true;
        }
    }
    else if (argc == 1) {
        printf("Running performance test for Conv2DLayerGPU with manual schedule.\n");
    }
    else {
        fprintf(stderr, "Usage: .//conv2d_gpu [autoscheduler]\n");
        fprintf(stderr, "       .//conv2d_gpu [--check/-c]\n");
        return 1;
    }

    const int batch_size = check ? 1 : 8, width = check ? 4: 120, height = check ? 4 : 100,  channels_in = check ? 2 : 3, channel_multiplier = check ? 2 : 2, kernel_size = check ? 3 : 5, stride = 1;

    try {
        // Generate random input. If we are in checking result mode, each entry = (sum of indices) * 0.1
        Buffer<float> input(channels_in, height, width, batch_size);
        for (int c = 0; c < channels_in; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int b = 0; b < batch_size; b++) {
                        if (check) {
                            input(c, h, w, b) = 0.1f * (c + h + w + b);
                        }
                        else {
                            input(c, h, w, b) = rand();
                        }
                    }
                }
            }
        }

        // Generate random filters. If we are in checking result mode, each entry = (sum of indices) * 0.1
        printf("Generating filters with dimensions: height: %d, width: %d, channels: %d, channels multiplier: %d\n", kernel_size, kernel_size, channels_in, channel_multiplier);

        Buffer<float> depthwise_filters(channel_multiplier, channels_in, kernel_size, kernel_size);
        for (int cm = 0; cm < channel_multiplier; cm++) {
            for (int ci = 0; ci < channels_in; ci++) {
                for (int x = 0; x < kernel_size; x++) {
                    for (int y = 0; y < kernel_size; y++) {
                        if (check) {
                            depthwise_filters(cm, ci, x, y) = 0.1f * (x + y + ci + cm);
                        }
                        else {
                            depthwise_filters(cm, ci, x, y) = rand();
                        }
                    }
                }
            }
        }

        printf("Running pipeline on GPU:\n");
        DepthwiseSepConv2DLayerCHWNGPU depth_conv_layer(input, depthwise_filters, stride, scheduler);
        printf("%s", scheduler.c_str());


        //conv_layer.schedule_for_gpu();
        if (check) {
            depth_conv_layer.print_result();
        }
        else {
            depth_conv_layer.test_performance(50);
        }
    }
    //Handle Halide exceptions
    catch (Halide::CompileError e) {
        printf(e.what());
        return -1;
    }
    catch (Halide::RuntimeError e) {
        printf(e.what());
        return -1;
    }
    catch (Halide::Error e) {
        printf(e.what());
        return -1;
    }

    return 0;
}

Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Target target = get_host_target();

    vector<Target::Feature> features_to_try;

    if (target.os == Target::OSX) {
        //For OSX, we will use Metal APIs
        features_to_try.push_back(Target::Metal);
    }
    else {
        //For Linux and Windows, use CUDA
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

    printf("Shape: (%d, %d, %d, %d)\n", b_hi, h_hi, w_hi, c_hi);
}
