#include "Flatten.h"
#include "../Types.h"
#include "../Utils.h"

namespace ML {

    void FlattenLayer::computeNaive(const LayerData& dataIn) const {
        // Retrieve the 3D array to flatten
        Array3D_fp32 dataIn3DArray = dataIn.getData<Array3D_fp32>();

        // Get dimensions of input data
        std::size_t inputHeight = getInputParams().dims[0];
        std::size_t inputWidth = getInputParams().dims[1];
        std::size_t inputColorChannels = getInputParams().dims[2];

        // Get the output data
        Array1D_fp32 dataOutArray = outData.getData<Array1D_fp32>();

        std::size_t idx = 0;
        for (std::size_t c = 0; c < inputColorChannels; ++c) {
            for (std::size_t h = 0; h < inputHeight; ++h) {
                for (std::size_t w = 0; w < inputWidth; ++w) {
                    // Flattening the 3D array to 1D
                    dataOutArray[idx++] = dataIn3DArray[h][w][c];
                }
            }
        }
    }

    void FlattenLayer::computeThreaded(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

    void FlattenLayer::computeTiled(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

    void FlattenLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};