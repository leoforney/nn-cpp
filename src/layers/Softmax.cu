#include "Softmax.h"

#include <iostream>
#include <cmath>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"


namespace ML {

    void SoftmaxLayer::computeNaive(const LayerData& dataIn) const {
        Array1D_fp32 dataInArray = dataIn.getData<Array1D_fp32>();
        std::size_t dataInSize = getInputParams().dims[0];

        Array1D_fp32 dataOutArray = outData.getData<Array1D_fp32>();
        std::size_t dataOutSize = getOutputParams().dims[0];

        if (dataInSize != dataOutSize) {
            throw std::invalid_argument("Input and output sizes do not match.");
        }

        auto maxInput = std::numeric_limits<fp32>::min();
        for (std::size_t i = 0; i < dataInSize; ++i) {
            if (dataInArray[i] > maxInput) {
                maxInput = dataInArray[i];
            }
        }

        auto sumExp = 0.0f;
        for (std::size_t i = 0; i < dataInSize; ++i) {
            dataOutArray[i] = std::exp(dataInArray[i] - maxInput);
            sumExp += dataOutArray[i];
        }

        for (std::size_t i = 0; i < dataInSize; ++i) {
            dataOutArray[i] /= sumExp;
        }
    }

    // Compute the convolution using SIMD
    void SoftmaxLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};  // namespace ML