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

void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
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

    std::vector<std::thread> threads;
    for (std::size_t f = 0; f < numFilters; f++) {
        threads.push_back(std::thread([=, &dataIn3DArray, &dataOut3DArray, &weightData4DArray]() {
            for (std::size_t out_y = 0; out_y < outHeight; out_y++) {
                for (std::size_t out_x = 0; out_x < outWidth; out_x++) {
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
        }));
    }
    for (auto& thread : threads) thread.join();
}

void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    static const std::size_t tileSize = 8;

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

    for (std::size_t f = 0; f < numFilters; f++) {
        for (std::size_t out_y_tile = 0; out_y_tile < outHeight; out_y_tile += tileSize) {
            for (std::size_t out_x_tile = 0; out_x_tile < outWidth; out_x_tile += tileSize) {
                for (std::size_t y = out_y_tile; y < std::min(out_y_tile + tileSize, outHeight); y++) {
                    for (std::size_t x = out_x_tile; x < std::min(out_x_tile + tileSize, outWidth); x++) {
                        float sum = 0.0f;
                        for (std::size_t dy = 0; dy < filterHeight; dy++) {
                            for (std::size_t dx = 0; dx < filterWidth; dx++) {
                                for (std::size_t c = 0; c < inputColorChannels; c++) {
                                    std::size_t in_y = y + dy;
                                    std::size_t in_x = x + dx;
                                    if (in_y < inputHeight && in_x < inputWidth) {
                                        sum += dataIn3DArray[in_y][in_x][c] * weightData4DArray[dy][dx][c][f];
                                    }
                                }
                            }
                        }
                        dataOut3DArray[y][x][f] = sum;
                    }
                }
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

        cudaMemcpy(d_dataIn, &dataIn3DArray[0][0][0], inputHeight * inputWidth * inputColorChannels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weightData, &weightData4DArray[0][0][0][0], filterHeight * filterWidth * numFilters * sizeof(float), cudaMemcpyHostToDevice);

        computeSimdKernel<<<numBlocks, threadsPerBlock>>>(d_dataIn, d_dataOut, d_weightData, numFilters, outHeight, outWidth, filterHeight, filterWidth, inputColorChannels, inputHeight, inputWidth);

        cudaMemcpy(&dataOut3DArray[0][0][0], d_dataOut, outHeight * outWidth * numFilters * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_dataIn);
        cudaFree(d_dataOut);
        cudaFree(d_weightData);
    }
};  // namespace ML