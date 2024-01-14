#include "Convolutional.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

// Compute the convolution for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
        Array3D_fp32 data = dataIn.getData<Array3D_fp32>();

        // Get the filter and output size details
        std::size_t filterHeight = getOutputParams().dims[0];
        std::size_t filterWidth = getOutputParams().dims[1];
        std::size_t numFilters = getOutputParams().dims[2];

        // Initialize output to the expected output size
        auto outDims = getOutputParams();

        // Loop over the input data
        for (int y = 0; y < outDims.dims[0]; y++) {
            for (int x = 0; x < outDims.dims[1]; x++) {
                for (int filter = 0; filter < numFilters; filter++){

                    // Initialize temporary sum for each filter
                    float sum = 0.0f;

                    // Apply the convolution
                    for (int ky = 0; ky < filterHeight; ky++) {
                        for (int kx = 0; kx < filterWidth; kx++) {
                            for (int colorChannel = 0; colorChannel < dataIn.getParams().dims[2]; colorChannel++) {

                                // Multiply filter value with corresponding input data value
                                // and add it to sum
                                sum += data[y + ky][x + kx][colorChannel] * weightData.getData<Array4D_fp32>()[filter][ky][kx][colorChannel];

                            }
                        }
                    }

                    // Add the bias of the current filter and apply an activation function (ReLU in this case)
                    outData.getData<Array3D_fp32>()[y][x][filter] = std::max(0.0f, sum + biasData.getData<Array1D_fp32>()[filter]);
                }
            }
        }
}

// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
};  // namespace ML