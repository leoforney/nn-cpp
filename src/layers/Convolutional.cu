#include "Convolutional.h"

#include <iostream>
#include <thread>
#include <vector>
#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

__global__ void computeSimdKernel(float* dataIn,
                                  float* dataOut,
                                  float* weightData,
                                  std::size_t numFilters,
                                  std::size_t outHeight,
                                  std::size_t outWidth,
                                  std::size_t filterHeight,
                                  std::size_t filterWidth,
                                  std::size_t inputColorChannels,
                                  std::size_t inputHeight,
                                  std::size_t inputWidth) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int f = index / (outHeight * outWidth);
    index %= outHeight * outWidth;
    int y = index / outWidth;
    int x = index % outWidth;

    if (f < numFilters && y < outHeight && x < outWidth) {
        float sum = 0.0f;
        for (std::size_t dy = 0; dy < filterHeight; dy++) {
            for (std::size_t dx = 0; dx < filterWidth; dx++) {
                for (std::size_t c = 0; c < inputColorChannels; c++) {
                    std::size_t in_y = y + dy;
                    std::size_t in_x = x + dx;
                    if (in_y < inputHeight && in_x < inputWidth) {
                        sum += dataIn[(in_y * inputWidth + in_x) * inputColorChannels + c] * weightData[(dy * filterWidth + dx) * numFilters + f];
                    }
                }
            }
        }
        dataOut[(y * outWidth + x) * numFilters + f] = sum;
    }
}

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
                    dataOut3DArray[out_y][out_x][f] = sum;
                }
            }
        }
    }

    void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
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

        dim3 threadsPerBlock(256);
        dim3 numBlocks((outHeight * outWidth * numFilters + threadsPerBlock.x - 1) / threadsPerBlock.x);

        float* d_dataIn;
        float* d_dataOut;
        float* d_weightData;
        cudaMalloc(&d_dataIn, inputHeight * inputWidth * inputColorChannels * sizeof(float));
        cudaMalloc(&d_dataOut, outHeight * outWidth * outColorChannels * sizeof(float));
        cudaMalloc(&d_weightData, filterHeight * filterWidth * numFilters * sizeof(float));

        cudaMemcpy(d_dataIn, &(dataIn3DArray[0][0][0]), inputHeight * inputWidth * inputColorChannels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weightData, &(weightData4DArray[0][0][0][0]), filterHeight * filterWidth * numFilters * sizeof(float), cudaMemcpyHostToDevice);

        computeSimdKernel<<<numBlocks, threadsPerBlock>>>(d_dataIn, d_dataOut, d_weightData, numFilters, outHeight, outWidth, filterHeight, filterWidth, inputColorChannels, inputHeight, inputWidth);

        float *h_dataOut = new float[outHeight * outWidth * numFilters];
        cudaMemcpy(h_dataOut, d_dataOut, outHeight * outWidth * numFilters * sizeof(float), cudaMemcpyDeviceToHost);

        for (std::size_t y = 0; y < outHeight; y++) {
            for (std::size_t x = 0; x < outWidth; x++) {
                for (std::size_t f = 0; f < numFilters; f++) {
                    dataOut3DArray[y][x][f] = h_dataOut[(y * outWidth + x) * numFilters + f];
                }
            }
        }

        cudaFree(d_dataIn);
        cudaFree(d_dataOut);
        cudaFree(d_weightData);

        delete[] h_dataOut;
    }
};  // namespace ML