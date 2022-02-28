#include <stdio.h>
#include <chrono>
#include <Magick++.h>
#include "utils.hpp"

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
        printf("adding images [%d]\n", ++j);
        TIMER_START(add image)
        // fastest access via pixels cache
        Pixels pixelsCache(image);
        MagickCore::Quantum *pixels = pixelsCache.get(0, 0, columns, rows);
        for (int i=0; i<rows*columns; i++) {
            averagedPixelsBuffer[i * 3] += pixels[i * 3];
            averagedPixelsBuffer[i * 3 + 1] += pixels[i * 3 + 1];
            averagedPixelsBuffer[i * 3 + 2] += pixels[i * 3 + 2];
        }
        TIMER_END(add image)
    }
    TIMER_START(ave image)
    const q n = (q) images.size();
    for (int i=0; i<rows*columns*3; i++) {
        averagedPixelsBuffer[i] /= n;
    }
    TIMER_END(ave image)
    Image aveImage(columns, rows, "RGB", Magick::StorageType::QuantumPixel, averagedPixelsBuffer);
    return aveImage;
    // q max = 0;
    // q min = 60000;
    // q ave = 0;
    // q nn = (q) rows*columns*3;
    // for (int i=0;i<rows*columns*3; i++) {
    //     max = std::max(max, averagedPixelsBuffer[i]);
    //     min = std::min(min, averagedPixelsBuffer[i]);
    //     ave += averagedPixelsBuffer[i] / nn;
    // }
    // printf("max: %f, min: %f, ave: %f\n", max, min, ave);
}

Image gpu_average(vector<Image> &images) {

}

int main(int argc, char **argv) {
    printf("size of quantum: %ld, pixelpacket: %ld\n", sizeof(MagickCore::Quantum), sizeof(MagickCore::PixelPacket));
    vector<string> imagePaths = glob("/home/zx02/Documents/Astro/Data/Other/M31 RAW 48F ISO800 8min/*.CR2");
    vector<Image> images;
    TIMER_START(load images)
    for (string path:imagePaths) {
        images.push_back(Image(path));
        break;
    }
    TIMER_END(load images)
    cpu_average(images);
    return 0;
}