#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
    class MaxPoolingLayer : public Layer {
    public:
        MaxPoolingLayer(const LayerParams inParams, const LayerParams outParams)
                : Layer(inParams, outParams, LayerType::MAX_POOLING) {}

        template <typename T> void allocateLayer() {
            Layer::allocateOutputBuffer<Array3D<T>>();
        }

        template <typename T> void freeLayer() {
            Layer::freeOutputBuffer<Array3D<T>>();
        }

        // Virtual functions
        virtual void computeNaive(const LayerData& dataIn) const override;
        virtual void computeSIMD(const LayerData& dataIn) const override;
    };

}  // namespace ML