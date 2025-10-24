#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>   // 1. For using std::vector
#include <cstring>  // 2. For using memcpy

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// 3. Change function signature:
//    We no longer pass in new_frontier, but a pointer to the "vector of local frontiers"
void top_down_step(Graph g, VertexSet *frontier,
                   std::vector<std::vector<int>> *local_frontiers,
                   int *distances)
{
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {
        // 4. Each thread gets its own ID
        int tid = omp_get_thread_num();
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                if (__sync_bool_compare_and_swap(&distances[outgoing],
                                                NOT_VISITED_MARKER,
                                                distances[node] + 1))
                {
                    // 5. (Core Optimization)
                    //    Remove the atomic 'fetch_and_add'
                    //    Instead, add the node to the *thread's own* local vector.
                    //    This is an asynchronous, contention-free operation.
                    (*local_frontiers)[tid].push_back(outgoing);
                }
            }
        }
    }
}

// Implements top-down BFS.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // 6. Create the "local frontiers" vector
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_frontiers(max_threads);

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        // 7. (Preparation) Clear all local frontiers in parallel
        #pragma omp parallel for
        for (int i = 0; i < max_threads; ++i)
        {
            local_frontiers[i].clear();
        }

        // 8. Call the modified top_down_step
        top_down_step(graph, frontier, &local_frontiers, sol->distances);

        // 9. (Sequential Merge Step)
        //    Merge the results from all local frontiers back into new_frontier
        vertex_set_clear(new_frontier);
        int current_offset = 0;
        for (int i = 0; i < max_threads; ++i)
        {
            int num_nodes_in_local = local_frontiers[i].size();
            if (num_nodes_in_local > 0)
            {
                // Use memcpy for efficient bulk memory copy
                memcpy(new_frontier->vertices + current_offset,
                       local_frontiers[i].data(),
                       num_nodes_in_local * sizeof(int));
                current_offset += num_nodes_in_local;
            }
        }
        new_frontier->count = current_offset;


#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}