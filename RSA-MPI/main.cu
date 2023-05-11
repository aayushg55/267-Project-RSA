#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>
#include <cuda.h>
#include "cublas_v2.h"
#include <curand.h>

using namespace std;
// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

void send_and_rec(double* d_send_buff, double* d_rec_buff, double** d_other_k, int rank, int count, int n_l, int n_r, bool even_iter) {
    if(even_iter) {
        // send from send_buff and recv new sub_k into recv
        if (rank % 2 == 0) {
            MPI_Send(d_send_buff, count, MPI_DOUBLE, n_r, 1, MPI_COMM_WORLD);
            MPI_Recv(d_rec_buff, count, MPI_DOUBLE, n_l, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(d_rec_buff, count, MPI_DOUBLE, n_l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(d_send_buff, count, MPI_DOUBLE, n_r, 2, MPI_COMM_WORLD); 
        }
        *d_other_k = d_rec_buff;
    } else {
        // send from recv_buff and recv new sub_k into send
        if (rank % 2 == 0) {
            MPI_Send(d_rec_buff, count, MPI_DOUBLE, n_r, 1, MPI_COMM_WORLD);
            MPI_Recv(d_send_buff, count, MPI_DOUBLE, n_l, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(d_send_buff, count, MPI_DOUBLE, n_l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(d_rec_buff, count, MPI_DOUBLE, n_r, 2, MPI_COMM_WORLD); 
        }
        *d_other_k = d_send_buff;
    }
}

MPI_Datatype PARTICLE;

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {

    // Init MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int NUM_ITER = find_int_arg(argc, argv, "-i", 100);
    bool do_backwards = find_int_arg(argc, argv, "-b", true);
    size_t seq_length = find_int_arg(argc, argv, "-n", 1440);
    size_t local_seq_length = ceil(seq_length/(num_procs));

    size_t qk_dim = 512;
    size_t v_dim = 512;

    size_t partition_qk_size = local_seq_length * qk_dim;
    size_t partition_v_size = local_seq_length * v_dim;
    size_t seq_seq = local_seq_length * local_seq_length;

    double* d_my_q;
    double* d_my_k;
    double* d_my_v;
    double* d_attn_scores;
    double* d_my_out;

    cudaMalloc((void**)&d_my_q, partition_qk_size * sizeof(double));
    cudaMalloc((void**)&d_my_k, partition_qk_size * sizeof(double));
    cudaMalloc((void**)&d_my_v, partition_v_size * sizeof(double));

    cudaMalloc((void**)&d_attn_scores, seq_seq * num_procs * sizeof(double));
    cudaMalloc((void**)&d_my_out, seq_seq * sizeof(double));

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, rank);
    
    curandGenerateUniformDouble(prng, d_my_q, partition_qk_size);
    curandGenerateUniformDouble(prng, d_my_k, partition_qk_size);
    curandGenerateUniformDouble(prng, d_my_v, partition_v_size);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alp = 1;
    const double bet = 0;
    const double *alpha = &alp;
    const double *beta = &bet;

    int n_r = (rank+1) % num_procs;
    int n_l = (rank-1+num_procs) % num_procs;

    double* d_rec_buff;
    double* d_send_buff;

    cudaMalloc((void**)&d_rec_buff, partition_qk_size * sizeof(double)); /* create array on GPU 0 */
    cudaMalloc((void**)&d_send_buff, partition_qk_size * sizeof(double)); /* create array on GPU 0 */
    MPI_Barrier(MPI_COMM_WORLD);

    auto start_time = std::chrono::steady_clock::now();
/*********************************************************************************************/
    // Ring QK

    for (int iter = 0; iter < NUM_ITER; iter++) {
        cudaMemcpy(d_send_buff, d_my_k, partition_qk_size, cudaMemcpyDeviceToDevice);
        // calculate own q * k_T
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, local_seq_length, qk_dim, alpha,
                        d_my_q, local_seq_length, 
                        d_my_k, local_seq_length, beta, 
                        d_attn_scores + seq_seq*rank, local_seq_length);
        cudaDeviceSynchronize();

        
        for (int i = 0; i < num_procs-1; i++) {
            double* d_other_k;
            send_and_rec(d_send_buff, d_rec_buff, &d_other_k, rank, partition_qk_size, n_l, n_r, (i+1)%2);

            //compute my_q * sub_k_t
            size_t neighbor_rank = (rank-i-1+num_procs) % num_procs;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, local_seq_length, qk_dim, alpha,
                        d_my_q, local_seq_length, 
                        d_other_k, local_seq_length, beta, 
                        d_attn_scores + seq_seq*neighbor_rank, local_seq_length);
            cudaDeviceSynchronize();    
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
/*********************************************************************************************/
    // compute softmax on A

/*********************************************************************************************/
    // Ring AV
    // copy my_v into send_buff
    for (int iter = 0; iter < NUM_ITER; iter++) {
        cudaMemcpy(d_send_buff, d_my_v, partition_v_size, cudaMemcpyDeviceToDevice);
        // calculate own A * V
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, v_dim, local_seq_length, alpha,
                d_attn_scores+seq_seq*rank, local_seq_length, 
                d_my_v, local_seq_length, beta, 
                d_my_out, local_seq_length);
        cudaDeviceSynchronize();

        for (int i = 0; i < num_procs-1; i++) {
            double* d_other_v;
            send_and_rec(d_send_buff, d_rec_buff, &d_other_v, rank, partition_v_size, n_l, n_r, (i+1)%2);

            //compute my_a * sub_v
            size_t neighbor_rank = (rank-i-1+num_procs) % num_procs;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, v_dim, local_seq_length, alpha,
                    d_attn_scores+seq_seq*neighbor_rank, local_seq_length, 
                    d_other_v, local_seq_length, alpha, 
                    d_my_out, local_seq_length);
            cudaDeviceSynchronize();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
/*********************************************************************************************/
if (do_backwards) {
    //backwards for AV
    // grad_upstream of dim local * v_dim; attn map of dim local * seq
    //m,n,k are post transpose

    double* d_grad_upstream;
    cudaMalloc((void**)&d_grad_upstream, local_seq_length*v_dim * sizeof(double));

    size_t grad_v_size = seq_length * v_dim;
    double* d_grad_v;
    cudaMalloc((void**)&d_grad_v, grad_v_size * sizeof(double));

    size_t grad_attn_scores_size = local_seq_length*seq_length;
    double* d_grad_attn_scores;
    cudaMalloc((void**)&d_grad_attn_scores, grad_attn_scores_size * sizeof(double));

    double* d_grad_q;
    cudaMalloc((void**)&d_grad_q, partition_qk_size * sizeof(double));

    size_t grad_k_size = seq_length * qk_dim;
    double* d_grad_k;
    cudaMalloc((void**)&d_grad_k, grad_k_size * sizeof(double));

    for (int j = 0; j < NUM_ITER; j++) {
        //compute grad_v = Attn_T * grad_upstream
        curandGenerateUniformDouble(prng, d_grad_upstream, local_seq_length*v_dim);
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_length, v_dim, local_seq_length, alpha,
                    d_attn_scores, local_seq_length, 
                    d_grad_upstream, local_seq_length, beta, 
                    d_grad_v, seq_length);
        cudaDeviceSynchronize();

        // reduce grad_v
        MPI_Allreduce(MPI_IN_PLACE, d_grad_v, grad_v_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        cudaMemcpy(d_send_buff, d_my_v, partition_v_size, cudaMemcpyDeviceToDevice);
        for (int i = 0; i < num_procs-1; i++) {
            double* d_other_v;
            send_and_rec(d_send_buff, d_rec_buff, &d_other_v, rank, partition_qk_size, n_l, n_r, (i+1)%2);

            //compute grad_upstream * sub_v_T
            size_t neighbor_rank = (rank-i-1+num_procs) % num_procs;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, local_seq_length, v_dim, alpha,
                    d_grad_upstream, local_seq_length, 
                    d_other_v, local_seq_length, alpha, 
                    d_grad_attn_scores + neighbor_rank*seq_seq, local_seq_length);
            cudaDeviceSynchronize();    
        }
        MPI_Barrier(MPI_COMM_WORLD);

    /*********************************************************************************************/
        //backward pass for QK_T

        // compute local grad_k contribution
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_length, qk_dim, local_seq_length, alpha,
                    d_grad_attn_scores, local_seq_length, 
                    d_my_q, local_seq_length, beta, 
                    d_grad_k, seq_length);

        //reduce grad_k over all processors
        MPI_Allreduce(MPI_IN_PLACE, d_grad_k, grad_k_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        cudaMemcpy(d_send_buff, d_my_k, partition_v_size, cudaMemcpyDeviceToDevice);
        for (int i = 0; i < num_procs-1; i++) {
            double* d_other_k;
            send_and_rec(d_send_buff, d_rec_buff, &d_other_k, rank, partition_qk_size, n_l, n_r, (i+1)%2);

            //compute grad_attn_scores[start:end] * sub_k
            size_t neighbor_rank = (rank-i-1+num_procs) % num_procs;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, qk_dim, local_seq_length, alpha,
                        d_grad_attn_scores + neighbor_rank*seq_seq, local_seq_length,
                        d_other_k, local_seq_length, alpha,
                        d_grad_q, local_seq_length);
            cudaDeviceSynchronize();    
        }
    }

    cudaFree(d_grad_upstream);
    cudaFree(d_grad_v);
    cudaFree(d_grad_k);
    cudaFree(d_grad_q);
    cudaFree(d_grad_attn_scores);
}
/*********************************************************************************************/
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    if (rank == 0) {
        cout << "Time = " << seconds << " seconds\n" << flush;
    }
    cudaFree(d_rec_buff);
    cudaFree(d_send_buff);
    cudaFree(d_my_q);
    cudaFree(d_my_k);
    cudaFree(d_my_v);
    cudaFree(d_attn_scores);
    cudaFree(d_my_out);


    MPI_Barrier(MPI_COMM_WORLD);
    // Finalize

    MPI_Finalize();
}
