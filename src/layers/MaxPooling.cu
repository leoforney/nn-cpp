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

        for (std::size_t c = 0; c < inputColorChannels; ++c) {
            for (std::size_t h = 0; h < inputHeight; h+=2) {
                for (std::size_t w = 0; w < inputWidth; w+=2) {
                    float maxPixel = -FLT_MAX;
                    for (std::size_t fh = 0; fh < 2; ++fh) {
                        for (std::size_t fw = 0; fw < 2; ++fw) {
                            float pixel = dataIn3DArray[h + fh][w + fw][c];
                            if (pixel > maxPixel) {
                                maxPixel = pixel;
                            }
                        }
                    }
                    dataOut3DArray[h / 2][w / 2][c] = maxPixel;
                }
            }
        }

    }

// Compute the convolution using SIMD
    void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};  // namespace ML