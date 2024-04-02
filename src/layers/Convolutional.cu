#include "Convolutional.h"

#include <iostream>
#include <thread>
#include <vector>
#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

    void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {

        Array3D_fp32 dataIn3DArray = dataIn.getData<Array3D_fp32>();
        std::size_t inputHeight = getInputParams().dims[0];
        std::size_t inputWidth = getInputParams().dims[1];
        std::size_t inputColorChannels = getInputParams().dims[2];

        Array3D_fp32 dataOut3DArray = outData.getData<Array3D_fp32>();
        auto outDims = getOutputParams();
        std::size_t outHeight = outDims.dims[0];
        std::size_t outWidth = outDims.dims[1];
        std::size_t outColorChannels = outDims.dims[2];

        std::size_t filterHeight = weightParam.dims[0];
        std::size_t filterWidth = weightParam.dims[1];
        std::size_t numFilters = weightParam.dims[3];

        Array4D<fp32> weightData4DArray = weightData.getData<Array4D<fp32>>();
        Array1D<fp32> biasDataArray = biasData.getData<Array1D<fp32>>();

        for (std::size_t out_y = 0; out_y < outHeight; out_y++) {
            for (std::size_t out_x = 0; out_x < outWidth; out_x++) {
                for (std::size_t f = 0; f < numFilters; f++) {
                    float sum = 0.0f;
                    for (std::size_t dy = 0; dy < filterHeight; dy++) {
                        for (std::size_t dx = 0; dx < filterWidth; dx++) {
                            for (std::size_t c = 0; c < inputColorChannels; c++) {
                                std::size_t in_y = out_y + dy;
                                std::size_t in_x = out_x + dx;
                                if (in_y < inputHeight && in_x < inputWidth) {
                                    sum += dataIn3DArray[in_y][in_x][c] * weightData4DArray[dy][dx][c][f];
                                }
                            }
                        }
                    }
                    sum += biasDataArray[f];
                    dataOut3DArray[out_y][out_x][f] = sum;
                }
            }
        }
    }

    void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {

    }
};  // namespace ML