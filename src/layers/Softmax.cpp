#include "Softmax.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

// Compute the convolution for the layer data
    void SoftmaxLayer::computeNaive(const LayerData& dataIn) const {

    }

// Compute the convolution using threads
    void SoftmaxLayer::computeThreaded(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

// Compute the convolution using a tiled approach
    void SoftmaxLayer::computeTiled(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }

// Compute the convolution using SIMD
    void SoftmaxLayer::computeSIMD(const LayerData& dataIn) const {
        // TODO: Your Code Here...
    }
};  // namespace ML