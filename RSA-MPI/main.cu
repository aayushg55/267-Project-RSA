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

MPI_Datatype PARTICLE;

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of particles" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", "");
    std::ofstream fsave(savename);

    // Init MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "num_procs: " << num_procs << "\n" << std::flush;
    // Create MPI Particle Type
    // const int nitems = 1;
    // int blocklengths[1] = {1};
    // MPI_Datatype types[7] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
    //                          MPI_DOUBLE,   MPI_DOUBLE, MPI_DOUBLE};
    // MPI_Aint offsets[7];
    // MPI_Type_create_struct(nitems, blocklengths, offsets, types, &PARTICLE);
    // MPI_Type_commit(&PARTICLE);

    // cudaStreamCreate(&stream_on_gpu_0);
    // cudaSetDevice(0);
    // cudaDeviceEnablePeerAccess( 1, 0 );
    // cudaMalloc(d_0, num_bytes); /* create array on GPU 0 */
    // cudaSetDevice(1);
    // cudaMalloc(d_1, num_bytes); /* create array on GPU 1 */
    // cudaMemcpyPeerAsync(d_0, 0, d_1, 1, num_bytes, stream_on_gpu_0);
    /* copy d_1 from GPU 1 to d_0 on GPU 0: pull copy */ 
    
    double ARR_SIZE = 2;
    double* rec_buff_d;
    double* send_buff_d;
    double* host = (double *) malloc(ARR_SIZE * sizeof(double));
    host[0] = rank;
    host[1] = -1;
    cudaMalloc((void**)&rec_buff_d, ARR_SIZE * sizeof(double)); /* create array on GPU 0 */
    cudaMalloc((void**)&send_buff_d, ARR_SIZE * sizeof(double)); /* create array on GPU 0 */
    cudaMemcpy(send_buff_d, host, ARR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "my rank: " << rank << "has host " << host[0] << "\n" << std::flush;

    // cudaMemcpy(rec_buff_d, host, 1 * sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "my rank: " << rank << "\n" << std::flush;

    for (int i = 0; i < num_procs-1; i++) {
        if(i % 2 == 0) {
            if (rank % 2 == 0) {
                MPI_Send(send_buff_d, ARR_SIZE, MPI_DOUBLE, (rank+1) % num_procs, 1, MPI_COMM_WORLD);
                MPI_Recv(rec_buff_d, ARR_SIZE, MPI_DOUBLE, (rank-1+num_procs) % num_procs, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(rec_buff_d, ARR_SIZE, MPI_DOUBLE, (rank+1) % num_procs, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buff_d, ARR_SIZE, MPI_DOUBLE, (rank-1+num_procs) % num_procs, 2, MPI_COMM_WORLD); 
            }
        } else {
            if (rank % 2 == 0) {
                MPI_Send(rec_buff_d, ARR_SIZE, MPI_DOUBLE, (rank+1) % num_procs, 1, MPI_COMM_WORLD);
                MPI_Recv(send_buff_d, ARR_SIZE, MPI_DOUBLE, (rank-1+num_procs) % num_procs, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(send_buff_d, ARR_SIZE, MPI_DOUBLE, (rank+1) % num_procs, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(rec_buff_d, ARR_SIZE, MPI_DOUBLE, (rank-1+num_procs) % num_procs, 2, MPI_COMM_WORLD); 
            }
        }
    }
    std::cout << rank << " Finished MPI sends!\n" << std::flush;
    std::cout << "my rank: " << rank << "before at 0 " << host[0] << "\n" << std::flush;
    std::cout << "my rank: " << rank << "before at 1 " << host[1] << "\n" << std::flush;
    cudaMemcpy(host, rec_buff_d, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "my rank: " << rank << "received at 0 " << host[0] << "\n" << std::flush;
    std::cout << "my rank: " << rank << "received at 1 " << host[1] << "\n" << std::flush;
    cudaFree(rec_buff_d);
    cudaFree(send_buff_d);
    free(host);
    // Initialize Particles
   
    // MPI_Bcast(parts, num_parts, PARTICLE, 0, MPI_COMM_WORLD);

    // Algorithm
    // auto start_time = std::chrono::steady_clock::now();

    // auto end_time = std::chrono::steady_clock::now();

    // std::chrono::duration<double> diff = end_time - start_time;
    // double seconds = diff.count();

    // Finalize
    std::cout << rank << " Finished RSA!\n" << std::flush;

    // if (rank == 1) {
    //     std::cout << "Init Time = " << init_time << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
    //     std::cout << "Apply force Time = " << apply_force_time << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
    //     std::cout << "Clear above/below grids = " << clear_grid_above_below << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
      
    //     std::cout << "Move Time = " << move_time << " seconds for " << num_parts
    //             << " particles.\n" << std::flush;
    //     std::cout << "Ghost send setup time = " << ghost_send << " seconds for " << num_parts
    //             << " particles.\n" << std::flush;
    //     std::cout << "Communication Time = " << comm << " seconds for " << num_parts
    //             << " particles.\n" << std::flush;    
    //     std::cout << "Clear send vectors Time = " << send_clear_time << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
    //     std::cout << "Rebin Time = " << rebin << " seconds for " << num_parts
    //               << " particles.\n\n" << std::flush;
    // }
    MPI_Finalize();
}