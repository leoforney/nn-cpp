#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
    class DenseLayer : public Layer {
    public:
        DenseLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, const LayerParams biasParams)
                : Layer(inParams, outParams, LayerType::DENSE),
                  weightParam(weightParams),
                  weightData(weightParams),
                  biasParam(biasParams),
                  biasData(biasParams) {}

        // Getters
        const LayerParams& getWeightParams() const { return weightParam; }
        const LayerParams& getBiasParams() const { return biasParam; }
        const LayerData& getWeightData() const { return weightData; }
        const LayerData& getBiasData() const { return biasData; }

        // Allocate all resources needed for the layer & Load all of the required data for the layer
        template <typename T> void allocateLayer() {
            Layer::allocateOutputBuffer<Array3D<T>>();
            weightData.loadData<Array4D<T>>();
            biasData.loadData<Array1D<T>>();
        }

        // Fre all resources allocated for the layer
        template <typename T> void freeLayer() {
            Layer::freeOutputBuffer<Array3D<T>>();
            weightData.freeData<Array4D<T>>();
            biasData.freeData<Array1D<T>>();
        }

        // Virtual functions
        virtual void computeNaive(const LayerData& dataIn) const override;
        virtual void computeSIMD(const LayerData& dataIn) const override;

    private:
        LayerParams weightParam;
        LayerData weightData;

        LayerParams biasParam;
        LayerData biasData;
    };

}  // namespace ML