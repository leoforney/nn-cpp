#include "Dense.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

// Compute the dense for the layer data
    void DenseLayer::computeNaive(const LayerData& dataIn) const {

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