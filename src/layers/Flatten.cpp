#include "Flatten.h"
#include "../Types.h"
#include "../Utils.h"

namespace ML {

    void FlattenLayer::computeNaive(const LayerData& dataIn) const {
        Array3D_fp32 dataIn3DArray = dataIn.getData<Array3D_fp32>();

        std::size_t inputHeight = getInputParams().dims[0];
        std::size_t inputWidth = getInputParams().dims[1];
        std::size_t inputColorChannels = getInputParams().dims[2];

        Array1D_fp32 dataOutArray = outData.getData<Array1D_fp32>();

        std::size_t idx = 0;
        for (std::size_t c = 0; c < inputColorChannels; ++c) {
            for (std::size_t h = 0; h < inputHeight; ++h) {
                for (std::size_t w = 0; w < inputWidth; ++w) {
                    dataOutArray[idx++] = dataIn3DArray[h][w][c];
                }
            }
        }
    }

    void FlattenLayer::computeSIMD(const LayerData& dataIn) const {
        computeNaive(dataIn);
    }
};