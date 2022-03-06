#include <stdio.h>
#include <chrono>
#include <Magick++.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "cuda/ops.h"

using namespace std;
using namespace Magick;

typedef MagickCore::Quantum q;

// HELPER
template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

auto start = chrono::steady_clock::now();

#define TIMER_START(label)\
start = chrono::steady_clock::now();\
printf(#label" [start]\n");

#define TIMER_END(label) printf(#label" [end]: %ldms\n", since(start).count());

Image cpu_average(vector<Image> &images) {
    int columns = images[0].columns();
    int rows = images[0].rows();
    
    auto *averagedPixelsBuffer = (q *) malloc(columns * rows * 3 * sizeof(q));
    memset(averagedPixelsBuffer, 0, columns * rows * 3 * sizeof(q));
    int j = 0;
    for (Image &image:images) {
        // fastest access via pixels cache
        Pixels pixelsCache(image);
        q *pixels = pixelsCache.get(0, 0, columns, rows);
        for (int i=0; i<rows*columns; i++) {
            averagedPixelsBuffer[i * 3] += pixels[i * 3];
            averagedPixelsBuffer[i * 3 + 1] += pixels[i * 3 + 1];
            averagedPixelsBuffer[i * 3 + 2] += pixels[i * 3 + 2];
        }
    }
    const q n = (q) images.size();
    for (int i=0; i<rows*columns*3; i++) {
        averagedPixelsBuffer[i] /= n;
    }
    Image aveImage(columns, rows, "RGB", Magick::StorageType::QuantumPixel, averagedPixelsBuffer);
    return aveImage;
}

Image gpu_average(vector<Image> &images) {
    int columns = images[0].columns();
    int rows = images[0].rows();

    TIMER_START(cpy imgs)
    q *u_imageBuffers;
    int imageSize = columns * rows * 3;
    cudaMallocManaged(&u_imageBuffers, images.size() * imageSize * sizeof(q));
    for (int i=0; i<images.size(); i++) {
        Pixels pixelsCache(images[i]);
        q *pixels = pixelsCache.get(0, 0, columns, rows);
        memcpy(u_imageBuffers + i*imageSize, pixels, imageSize);
    }
    TIMER_END(cpy imgs)
    TIMER_START(ave imgs)
    op_reduce_average_2d(u_imageBuffers, images.size(), imageSize);
    cudaDeviceSynchronize();
    TIMER_END(ave_imgs)
}

int main(int argc, char **argv) {
    printf("size of quantum: %ld, pixelpacket: %ld\n", sizeof(q), sizeof(MagickCore::PixelPacket));
    vector<string> imagePaths = glob("/home/zx02/Documents/Astro/Data/Other/M31 RAW 48F ISO800 8min/*.CR2");
    vector<Image> images;
    TIMER_START(load images)
    for (string path:imagePaths) {
        images.push_back(Image(path));
    }
    TIMER_END(load images)
    TIMER_START(cpu ave)
    Image image = cpu_average(images);
    TIMER_END(cpu ave)
    // image.write("average.jpeg");
    gpu_average(images);
}