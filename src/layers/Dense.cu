#include "Dense.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

// Compute the dense for the layer data
    void DenseLayer::computeNaive(const LayerData& dataIn) const {
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

        for(std::size_t filter = 0; filter < numFilters; ++filter) {
            for (std::size_t oRow = 0; oRow < outHeight; ++oRow){
                for (std::size_t oCol = 0; oCol < outWidth; ++oCol){
                    fp32 sum = 0.0;
                    for (std::size_t iChan = 0; iChan < inputColorChannels; ++iChan){
                        for (std::size_t kRow = 0; kRow < filterHeight; ++kRow){
                            for (std::size_t kCol = 0; kCol < filterWidth; ++kCol){
                                sum += dataIn3DArray[oRow + kRow][oCol + kCol][iChan]
                                       * weightData4DArray[kRow][kCol][iChan][filter];
                            }
                        }
                    }
                    dataOut3DArray[oRow][oCol][filter] = sum;
                }
            }
        }
    }

// Compute the dense using threads
    void DenseLayer::computeThreaded(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

// Compute the dense using a tiled approach
    void DenseLayer::computeTiled(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

// Compute the dense using SIMD
    void DenseLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};  // namespace ML