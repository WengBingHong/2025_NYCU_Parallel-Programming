#include <mpi.h>
#include <cstring> // For memcpy, memset
#include <algorithm> // For std::min

// --- 1D Row-wise Tiled Solution ---
//
// 通訊策略 (construct_matrices):
// 1. A (n x m) 矩陣：按「列」(row) 切割成 p 塊。
//    Rank 0 使用 MPI_Scatterv 將每個 A_local 區塊 (local_n x m) 發送給對應的 process。
// 2. B (m x l) 矩陣：保持完整。
//    Rank 0 使用 MPI_Bcast 將 **完整的 B 矩陣** 廣播給所有 process。
//
// 計算策略 (matrix_multiply):
// 1. 每個 process 都有 A_local (local_n x m) 和 B_full (m x l)。
// 2. 每個 process 計算 C_local (local_n x l) = A_local * B_full。
// 3. 效能優化
//    本地計算 C_local = A_local * B_full 時，使用 Tiling/Blocking。
//    我們不使用 (i, j, k) 三層迴圈，而是使用 (ii, jj, kk, i, j, k) 六層迴圈，
//    將計算切成 B_SIZE x B_SIZE 的小區塊，最大化 CPU L1/L2 快取命中率。
//    這能顯著提升本地計算速度。
// 4. Rank 0 使用 MPI_Gatherv 從所有 process 收集 C_local 區塊，組合成最終的 C 矩陣。
//
// 記憶體用量:
// - 每個 Process: O((n*m)/p + (m*l))
// - 總和: O(n*m + p*m*l) -> B 矩陣的儲存是效能瓶頸，但本地計算很快。

// 輔助函式：計算每個 process 應分得的 row 數量和起始 offset
// 這是我們在 1D row-wise 分解中需要的
inline void get_distribution(int n, int p, int rank, int& local_n, int& offset) {
    int rows_per_proc = n / p;
    int remainder = n % p;
    if (rank < remainder) {
        local_n = rows_per_proc + 1;
        offset = rank * local_n;
    } else {
        local_n = rows_per_proc;
        offset = rank * rows_per_proc + remainder;
    }
}

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. 分配和分發 Matrix A (Row-wise partitioning)
    
    int *sendcounts = nullptr;
    int *displs = nullptr;
    
    // 計算此 process 應有的 local_n (rows) 和 offset (starting row)
    int local_n, offset;
    get_distribution(n, world_size, rank, local_n, offset);
    
    int local_size_a = local_n * m;
    *a_mat_ptr = new int[local_size_a];

    if (rank == 0) {
        sendcounts = new int[world_size];
        displs = new int[world_size];
        
        for (int i = 0; i < world_size; ++i) {
            int proc_rows, proc_offset;
            get_distribution(n, world_size, i, proc_rows, proc_offset);
            
            sendcounts[i] = proc_rows * m; // 每個 A 區塊的大小
            displs[i] = proc_offset * m;   // 每個 A 區塊的起始位移
        }
    }

    // 將 A 矩陣的各個 row-block 從 rank 0 Scatterv 出去
    MPI_Scatterv(a_mat,         // (root only) send buffer
                 sendcounts,    // (root only)
                 displs,        // (root only)
                 MPI_INT,
                 *a_mat_ptr,    // receive buffer
                 local_size_a,  // receive count
                 MPI_INT,
                 0,             // root
                 MPI_COMM_WORLD);

    if (rank == 0) {
        delete[] sendcounts;
        delete[] displs;
    }

    // 2. 分配和廣播 Matrix B (Broadcast entire matrix)
    
    int size_b = m * l;
    *b_mat_ptr = new int[size_b];

    // Rank 0 需要從 main.cc 傳入的 b_mat 複製
    if (rank == 0) {
        std::memcpy(*b_mat_ptr, b_mat, size_b * sizeof(int));
    }

    // 將 B 矩陣從 rank 0 廣播給所有 process
    MPI_Bcast(*b_mat_ptr, size_b, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. 獲取此 process 負責的 row 數量
    int local_n, offset;
    get_distribution(n, world_size, rank, local_n, offset);

    // 2. 分配本地 C 矩陣 (C_local)
    int local_size_c = local_n * l;
    int *local_out_mat = new int[local_size_c];
    
    // 效能優化：Tiled (Blocked) Matrix Multiplication
    //
    // a_mat (A_local) 是 (local_n x m) row-major
    // b_mat (B_full)  是 (m x l) column-major (由 main.cc 讀取方式決定)
    //
    // 我們要計算 C(i, j) = sum_k A(i, k) * B(k, j)
    //
    // 最佳的本地迴圈 (i, j, k) 順序：
    // for i: C(i,j) 和 A(i,k) 的 access pattern 良好
    //   for j: C(i,j) 和 B(k,j) 的 access pattern 良好
    //     for k: A(i,k) [stride 1, good] 和 B(k,j) [stride 1, good]
    //
    // 我們在 (i, j, k) 迴圈之上再加入 (ii, jj, kk) Tiling 迴圈
    // 來最大化快取重用 (cache reuse)

    // 選擇一個 block size (e.g., 32 或 64)。這可以作為一個調整參數
    // 32 * 32 * sizeof(int) (約 4KB) 很容易放進 L1 data cache
    const int B_SIZE = 32; 

    // 將 C_local 初始化為 0
    std::memset(local_out_mat, 0, local_size_c * sizeof(int));

    // 六層迴圈的 Tiled / Blocked 矩陣乘法
    for (int ii = 0; ii < local_n; ii += B_SIZE) {
        for (int jj = 0; jj < l; jj += B_SIZE) {
            for (int kk = 0; kk < m; kk += B_SIZE) {
                
                // 計算 C(ii:ii+B, jj:jj+B) += A(ii:ii+B, kk:kk+B) * B(kk:kk+B, jj:jj+B)
                
                // 確定此 tile 的邊界 (避免超出矩陣)
                int i_end = std::min(ii + B_SIZE, local_n);
                int j_end = std::min(jj + B_SIZE, l);
                int k_end = std::min(kk + B_SIZE, m);

                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        // 在最內層迴圈計算 dot product
                        // C[i][j] = C[i][j] + sum(A[i][k] * B[k][j])
                        int sum = local_out_mat[i * l + j]; // 載入 (kk=0 時為 0)
                        
                        for (int k = kk; k < k_end; ++k) {
                            // a_mat[i * m + k] -> A_local(i, k) [row-major]
                            // b_mat[k + j * m] -> B_full(k, j)  [col-major]
                            // 兩個 access 都是 stride-1 (連續存取), 效能極佳
                            sum += a_mat[i * m + k] * b_mat[k + j * m];
                        }
                        local_out_mat[i * l + j] = sum; // 寫回
                    }
                }
            }
        }
    }
    // *** Tiled (Blocked) Matrix Multiplication 結束 ***


    // 3. 將所有 C_local 區塊 Gather 回 rank 0
    
    int *recvcounts = nullptr;
    int *displs = nullptr;

    if (rank == 0) {
        recvcounts = new int[world_size];
        displs = new int[world_size];
        
        for (int i = 0; i < world_size; ++i) {
            int proc_rows, proc_offset;
            get_distribution(n, world_size, i, proc_rows, proc_offset);
            
            recvcounts[i] = proc_rows * l; // 每個 C 區塊的大小
            displs[i] = proc_offset * l;   // 每個 C 區塊的起始位移
        }
    }

    // 將所有 local_out_mat (C_local) 收集到 rank 0 的 out_mat (C_full)
    MPI_Gatherv(local_out_mat, // send buffer
                local_size_c,  // send count
                MPI_INT,
                out_mat,       // (root only) receive buffer
                recvcounts,    // (root only)
                displs,        // (root only)
                MPI_INT,
                0,             // root
                MPI_COMM_WORLD);

    // 4. 清理本地記憶體
    delete[] local_out_mat;
    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    // 釋放 construct_matrices 中 new 出來的記憶體
    delete[] a_mat;
    delete[] b_mat;
}

// #include <mpi.h>
// #include <cstring> // For memcpy

// // Helper function to calculate row distribution
// inline void get_distribution(int n, int p, int rank, int& local_n, int& offset) {
//     int rows_per_proc = n / p;
//     int remainder = n % p;
//     if (rank < remainder) {
//         local_n = rows_per_proc + 1;
//         offset = rank * local_n;
//     } else {
//         local_n = rows_per_proc;
//         offset = rank * rows_per_proc + remainder;
//     }
// }

// void construct_matrices(
//     int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
// {
//     int rank, world_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     // 1. Distribute Matrix A (Row-wise partitioning)
    
//     int *sendcounts = nullptr;
//     int *displs = nullptr;
    
//     // Calculate local_n (rows for this process) and offset (starting row)
//     int local_n, offset;
//     get_distribution(n, world_size, rank, local_n, offset);
    
//     int local_size_a = local_n * m;
//     *a_mat_ptr = new int[local_size_a];

//     if (rank == 0) {
//         sendcounts = new int[world_size];
//         displs = new int[world_size];
        
//         int current_offset_rows = 0;
//         for (int i = 0; i < world_size; ++i) {
//             int proc_rows, proc_offset;
//             get_distribution(n, world_size, i, proc_rows, proc_offset);
            
//             sendcounts[i] = proc_rows * m;
//             displs[i] = proc_offset * m;
//         }
//     }

//     // Scatter the rows of A from rank 0 to all processes
//     MPI_Scatterv(a_mat,         // Send buffer (only used by root)
//                  sendcounts,    // Array of send counts
//                  displs,        // Array of displacements
//                  MPI_INT,       // Send type
//                  *a_mat_ptr,    // Receive buffer
//                  local_size_a,  // Receive count
//                  MPI_INT,       // Receive type
//                  0,             // Root process
//                  MPI_COMM_WORLD);

//     if (rank == 0) {
//         delete[] sendcounts;
//         delete[] displs;
//     }

//     // 2. Distribute Matrix B (Broadcast entire matrix)
    
//     int size_b = m * l;
//     *b_mat_ptr = new int[size_b];

//     if (rank == 0) {
//         // Rank 0 copies from the original b_mat to its new b_mat_ptr
//         std::memcpy(*b_mat_ptr, b_mat, size_b * sizeof(int));
//     }

//     // Broadcast B from rank 0 to all other processes
//     MPI_Bcast(*b_mat_ptr, size_b, MPI_INT, 0, MPI_COMM_WORLD);
// }

// void matrix_multiply(
//     const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
// {
//     int rank, world_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     // 1. Calculate local portion of C
    
//     // Get this process's number of rows
//     int local_n, offset;
//     get_distribution(n, world_size, rank, local_n, offset);

//     int local_size_c = local_n * l;
//     int *local_out_mat = new int[local_size_c];

//     // Perform local matrix multiplication
//     // a_mat is (local_n x m) row-major
//     // b_mat is (m x l) column-major (as read by main.cc)
//     // local_out_mat is (local_n x l) row-major
//     for (int i = 0; i < local_n; ++i) { // For each local row of A
//         for (int j = 0; j < l; ++j) {     // For each column of B
//             int sum = 0;
//             for (int k = 0; k < m; ++k) { // Dot product
//                 // a_mat[i*m + k] is A_local(i, k)
//                 // b_mat[j*m + k] is B_full(k, j)
//                 sum += a_mat[i * m + k] * b_mat[j * m + k];
//             }
//             local_out_mat[i * l + j] = sum;
//         }
//     }

//     // 2. Gather all local_out_mat blocks to rank 0
    
//     int *recvcounts = nullptr;
//     int *displs = nullptr;

//     if (rank == 0) {
//         recvcounts = new int[world_size];
//         displs = new int[world_size];
        
//         for (int i = 0; i < world_size; ++i) {
//             int proc_rows, proc_offset;
//             get_distribution(n, world_size, i, proc_rows, proc_offset);
            
//             recvcounts[i] = proc_rows * l;
//             displs[i] = proc_offset * l;
//         }
//     }

//     // Gather the results from all processes to out_mat on rank 0
//     MPI_Gatherv(local_out_mat, // Send buffer
//                 local_size_c,  // Send count
//                 MPI_INT,       // Send type
//                 out_mat,       // Receive buffer (only used by root)
//                 recvcounts,    // Array of receive counts
//                 displs,        // Array of displacements
//                 MPI_INT,       // Receive type
//                 0,             // Root process
//                 MPI_COMM_WORLD);

//     // 3. Clean up local memory
//     delete[] local_out_mat;
//     if (rank == 0) {
//         delete[] recvcounts;
//         delete[] displs;
//     }
// }

// void destruct_matrices(int *a_mat, int *b_mat)
// {
//     // Free the memory allocated in construct_matrices
//     delete[] a_mat;
//     delete[] b_mat;
// }