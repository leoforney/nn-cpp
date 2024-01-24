#include "Softmax.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

    void SoftmaxLayer::computeNaive(const LayerData& dataIn) const {
        Array3D_fp32 dataIn3DArray = dataIn.getData<Array3D_fp32>();
        std::size_t inputHeight = getInputParams().dims[0];
        std::size_t inputWidth = getInputParams().dims[1];
        std::size_t inputColorChannels = getInputParams().dims[2];

        Array3D_fp32 dataOut3DArray = outData.getData<Array3D_fp32>();
        auto outDims = getOutputParams();
        std::size_t outHeight = outDims.dims[0];
        std::size_t outWidth = outDims.dims[1];
        std::size_t outColorChannels = outDims.dims[2];

        for (std::size_t i = 0; i < outHeight; ++i) {
            for (std::size_t j = 0; j < outWidth; ++j) {
                for (std::size_t c = 0; c < outColorChannels; ++c) {
                    float exp_val = std::exp(dataIn3DArray[i][j][c]);
                    dataOut3DArray[i][j][c] = exp_val;
                }
            }
        }

        for (std::size_t i = 0; i < outHeight; ++i) {
            for (std::size_t j = 0; j < outWidth; ++j) {
                float sum_exp = 0.0;
                for (std::size_t c = 0; c < outColorChannels; ++c) {
                    sum_exp += dataOut3DArray[i][j][c];
                }

                for (std::size_t c = 0; c < outColorChannels; ++c) {
                    dataOut3DArray[i][j][c] /= sum_exp;
                }
            }
        }

    }

    // Compute the convolution using SIMD
    void SoftmaxLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};  // namespace ML