#include "Dense.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

__global__ void matmulKernel(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void computeMatMul(const thrust::device_vector<float>& A, const thrust::device_vector<float>& B, thrust::device_vector<float>& C, int M, int K, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(A.data()),
                                                     thrust::raw_pointer_cast(B.data()),
                                                     thrust::raw_pointer_cast(C.data()),
                                                     M, K, N);

    cudaDeviceSynchronize();
}

namespace ML {
    void DenseLayer::computeNaive(const LayerData& dataIn) const {
        Array1D_fp32 dataInArray = dataIn.getData<Array1D_fp32>();
        std::size_t inputSize = getInputParams().dims[0];

        Array1D_fp32 dataOutArray = outData.getData<Array1D_fp32>();
        std::size_t outSize = getOutputParams().dims[0];

        Array1D<fp32> biasDataArray = biasData.getData<Array1D<fp32>>();
        std::size_t biasSize = biasParam.dims[0];

        Array2D<fp32> weightDataArray = weightData.getData<Array2D<fp32>>();
        std::size_t weightDimX = weightParam.dims[0];
        std::size_t weightDimY = weightParam.dims[1];

        if (inputSize != weightDimY || outSize != weightDimX || outSize != biasSize) {
            throw std::invalid_argument("Dimension mismatch in DenseLayer::computeNaive");
        }

        for (std::size_t i = 0; i < outSize; ++i) {
            dataOutArray[i] = 0.0f;
        }

        for (std::size_t i = 0; i < outSize; ++i) {
            for (std::size_t j = 0; j < inputSize; ++j) {
                dataOutArray[i] += dataInArray[j] * weightDataArray[i][j];
            }
        }

        for (std::size_t i = 0; i < outSize; ++i) {
            dataOutArray[i] += biasDataArray[i];
        }
    }

    void DenseLayer::computeSIMD(const LayerData& dataIn) const {
        std::size_t M = 1;
        std::size_t K = getInputParams().dims[0];
        std::size_t N = getOutputParams().dims[0];

        thrust::device_vector<float> d_A(dataIn.getData<Array1D_fp32>(), dataIn.getData<Array1D_fp32>() + K);
        thrust::device_vector<float> d_B(weightData.getData<Array2D_fp32>(), weightData.getData<Array2D_fp32>() + K * N);
        thrust::device_vector<float> d_C(N);
        thrust::device_vector<float> d_bias(biasData.getData<Array1D<fp32>>(), biasData.getData<Array1D<fp32>>() + N);

        computeMatMul(d_A, d_B, d_C, M, K, N);

        for (size_t i = 0; i < N; ++i) {
            thrust::transform(d_C.begin() + i, d_C.begin() + i + 1, d_bias.begin() + i, d_C.begin() + i, thrust::plus<float>());
        }

        thrust::copy(d_C.begin(), d_C.end(), outData.getData<Array1D_fp32>());
    }

};  // namespace ML