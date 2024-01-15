#include "MaxPooling.h"

#include <iostream>
#include <cfloat>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"


namespace ML {

// Compute the convolution for the layer data
    void MaxPoolingLayer::computeNaive(const LayerData& dataIn) const {
        Array3D_fp32 dataIn3DArray = dataIn.getData<Array3D_fp32>();
        std::size_t inputHeight = getInputParams().dims[0];
        std::size_t inputWidth = getInputParams().dims[1];
        std::size_t inputColorChannels = getInputParams().dims[2];

        Array3D_fp32 dataOut3DArray = outData.getData<Array3D_fp32>();
        auto outDims = getOutputParams();
        std::size_t outHeight = outDims.dims[0];
        std::size_t outWidth = outDims.dims[1];
        std::size_t outColorChannels = outDims.dims[2];

        for (std::size_t i = 0; i < 2 * outHeight; i += 2) {
            for (std::size_t j = 0; j < 2 * outWidth; j += 2) {

                for (std::size_t c = 0; c < inputColorChannels; ++c) {

                    float maxVal = -FLT_MAX;

                    for (int h = 0; h < 2; ++h) {
                        for (int w = 0; w < 2; ++w) {
                            // Ensure that we don't access beyond the input dimensions
                            if ((i + h) < inputHeight && (j + w) < inputWidth) {
                                maxVal = std::max(maxVal, dataIn3DArray[i+h][j+w][c]);
                            }
                        }
                    }

                    dataOut3DArray[i/2][j/2][c] = maxVal;
                }
            }
        }
    }

// Compute the convolution using threads
    void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

// Compute the convolution using a tiled approach
    void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

// Compute the convolution using SIMD
    void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};  // namespace ML