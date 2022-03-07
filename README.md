# NVAstro

This project aims to replicate the functions of [DeepSkyStacker](http://deepskystacker.free.fr/english/index.html), speeding it up by offloading work to a CUDA-capable GPU whenever appropriate.

---
## Functions
Below's the checklist of functions to be implemented, roughly in order of their application according to DSS.

- Alignment
    - Star detection
    - Offset & angle computation
    - Applying transformation
- Stacking
    - Background calibration
    - Flat frame calibration
    - Hot pixel detection & removal
    - [Low Priority] Removal of bad columns
    - [Low Priority] Entropy based dark frame subtraction
    - Stacking [WIP]
        - Creating master offset, dark, flat frames
        - Methods:
            - Average
            - Median

---
## Project Structure
- `.vscode` VSCode workspace settings.
- `build` Built products & intermediates.
- `src` Source files.
    - `cuda` The cuda code used in nvastro.
    - `main.cpp` The nvastro application's entrypoint.
    - `test_cuda.cu` The test_cuda target's entrypoint. It's linked with codes in the `cuda` folder, allowing stuff there to be tested without going through the main app.
- `misc.ipynb` Random experiments.

---
## Compilation
Compiling this project requires the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit), g++ compiler, [LibRaw](https://www.libraw.org/about) & [ImageMagick](https://imagemagick.org/index.php) libraries.

In the root folder of this project, run:

`make nvastro`

Or if you want to make and run the `test_cuda` target:

`make test_cuda`