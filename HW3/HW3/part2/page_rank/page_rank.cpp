#include "page_rank.h"

#include <cmath> // Needed for std::abs
#include <cstdlib>
#include <omp.h>
#include <vector>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{
    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;

    // Parallelize the initialization loop
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    /*
       (Pseudocode comments omitted for brevity)
     */

    // 1. Allocate a temporary array to store 'score_new'
    //    Using std::vector for automatic memory management (RAII)
    std::vector<double> score_new(nnodes);
    bool converged = false;

    // 3. Pre-calculate the constant base contribution
    const double base_contrib = (1.0 - damping) / nnodes;

    while (!converged)
    {
        // --- Pseudocode Steps 1 & 2: Handle Dangling Nodes ---
        double dangling_sum = 0.0;
        
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int v = 0; v < nnodes; ++v)
        {
            if (outgoing_size(g, v) == 0)
            {
                // The solution array acts as score_old at this point
                dangling_sum += solution[v];
            }
        }
        
        double dangling_contrib = damping * dangling_sum / nnodes;

        // --- Pseudocode Step 1: Compute score_new ---
        #pragma omp parallel for
        for (int vi = 0; vi < nnodes; ++vi)
        {
            double sum_from_neighbors = 0.0;

            const Vertex* start = incoming_begin(g, vi);
            const Vertex* end = incoming_end(g, vi);

            for (const Vertex* vj_ptr = start; vj_ptr != end; ++vj_ptr)
            {
                int vj = *vj_ptr;
                int num_outgoing_vj = outgoing_size(g, vj);
                
                if (num_outgoing_vj > 0)
                {
                    sum_from_neighbors += solution[vj] / num_outgoing_vj;
                }
            }

            // Apply damping factor and pre-calculated base probability
            score_new[vi] = (damping * sum_from_neighbors) + base_contrib;

            // Add the contribution from all dangling nodes
            score_new[vi] += dangling_contrib;
        }


        // --- Pseudocode Step 3 & 4: Check convergence AND Prepare for next iteration ---
        //    (Loops are fused into one pass)
        
        double global_diff = 0.0;
        
        #pragma omp parallel for reduction(+:global_diff)
        for (int vi = 0; vi < nnodes; ++vi)
        {
            // Step 3: Check for convergence
            global_diff += std::abs(score_new[vi] - solution[vi]);
            
            // Step 4: Prepare for next iteration (Copy)
            solution[vi] = score_new[vi];
        }
        
        converged = (global_diff < convergence);

    } // end while(!converged)

    // 5. No need to 'delete[] score_new', std::vector handles it automatically
}