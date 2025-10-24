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

// -------------------------------------------------------------------
// Task 2: Top-Down (Optimized Version from previous step)
// -------------------------------------------------------------------

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

// -------------------------------------------------------------------
// Task 3: Bottom-Up (Optimized Implementation)
// -------------------------------------------------------------------

void bottom_up_step(Graph g,
                    std::vector<std::vector<int>> *local_frontiers,
                    int *distances,
                    int current_distance)
{
    // 1. (Core) Parallelize the loop over *all* nodes v
    //    *** Performance Tuning: ***
    //    Add schedule(dynamic, 768)
    //    This lets threads dynamically claim chunks of work,
    //    to handle the load imbalance common in real-world graphs
    //    (e.g., some nodes have far more 'incoming edges' than others)
    #pragma omp parallel for schedule(dynamic, 768)
    for (int v = 0; v < g->num_nodes; v++)
    {
        // 2. Check if 'v' is not visited
        if (distances[v] == NOT_VISITED_MARKER)
        {
            // 3. Iterate over 'v's *incoming* neighbors 'u'
            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[v + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int u = g->incoming_edges[neighbor];

                // 4. Check if neighbor 'u' is in the *previous* level (i.e., the `frontier`)
                if (distances[u] == current_distance)
                {
                    // 5. (Sync) Found it! Set 'v's distance.
                    //    (No CAS needed)
                    distances[v] = current_distance + 1;

                    // 6. (Sync) Add 'v' to the thread-local frontier
                    int tid = omp_get_thread_num();
                    (*local_frontiers)[tid].push_back(v);

                    // 7. (Optimization) Found one, break from the inner loop
                    break;
                }
            }
        }
    }
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    // (This function's content is identical to the previous version, no changes needed)
    // 1. Setup VertexSet (same as Top-Down)
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // 2. Setup thread-local frontiers (same as Top-Down)
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_frontiers(max_threads);

    // 3. Initialize distances array (same as Top-Down)
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // 4. Setup root node (same as Top-Down)
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // 5. Keep track of the current BFS "level"
    int current_distance = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif
        // 6. Clear local frontiers in parallel (same as Top-Down)
        #pragma omp parallel for
        for (int i = 0; i < max_threads; ++i)
        {
            local_frontiers[i].clear();
        }

        // 7. (Core) Call *bottom_up_step*
        bottom_up_step(graph, &local_frontiers, sol->distances, current_distance);

        // 8. Sequential merge (same as Top-Down)
        vertex_set_clear(new_frontier);
        int current_offset = 0;
        for (int i = 0; i < max_threads; ++i)
        {
            int num_nodes_in_local = local_frontiers[i].size();
            if (num_nodes_in_local > 0)
            {
                memcpy(new_frontier->vertices + current_offset,
                       local_frontiers[i].data(),
                       num_nodes_in_local * sizeof(int));
                current_offset += num_nodes_in_local;
            }
        }
        new_frontier->count = current_offset;


#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        // Print *new* frontier size
        printf("frontier=%-10d %.4f sec\n", new_frontier->count, end_time - start_time);
#endif

        // 9. Swap pointers (same as Top-Down)
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        // 10. (New) Increment to the next level
        current_distance++;
    }

    // 11. Cleanup (same as Top-Down)
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

// -------------------------------------------------------------------
// Task 4: Hybrid BFS (New Implementation)
// -------------------------------------------------------------------

void bfs_hybrid(Graph graph, solution *sol)
{
    // 1. Setup VertexSet (same as Top-Down / Bottom-Up)
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // 2. Setup thread-local frontiers (same as Top-Down / Bottom-Up)
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_frontiers(max_threads);

    // 3. Initialize distances array (same as Top-Down / Bottom-Up)
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // 4. Setup root node (same as Top-Down / Bottom-Up)
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // 5. Keep track of the current BFS "level" (same as Bottom-Up)
    int current_distance = 0;

    // 6. (Hybrid Strategy) Heuristics parameters
    //    These parameters are from the "Direction-Optimizing Breadth-First Search" paper
    //    We switch to Bottom-Up when the frontier gets too large
    //    and switch back to Top-Down when it becomes small again
    const int num_nodes = graph->num_nodes;
    const int threshold_td_to_bu = num_nodes / 24; // (alpha) Threshold to switch to BU
    const int threshold_bu_to_td = num_nodes / 12; // (beta)  Threshold to switch back to TD

    // 7. (Hybrid Strategy) Track current mode, start with Top-Down
    bool use_top_down = true;


    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif
        // 8. Clear local frontiers in parallel (same as Top-Down / Bottom-Up)
        #pragma omp parallel for
        for (int i = 0; i < max_threads; ++i)
        {
            local_frontiers[i].clear();
        }

        // 9. (Core) !!! Hybrid Strategy Decision !!!
        if (use_top_down)
        {
            // (a) Currently in Top-Down mode

            // Call your Task 2
            top_down_step(graph, frontier, &local_frontiers, sol->distances);

            // (b) Sequential merge (same as Top-Down)
            //     Note: We must merge *before* we can get 'new_frontier->count'
            //     to make the decision for the next iteration
            vertex_set_clear(new_frontier);
            int current_offset = 0;
            for (int i = 0; i < max_threads; ++i)
            {
                int num_nodes_in_local = local_frontiers[i].size();
                if (num_nodes_in_local > 0)
                {
                    memcpy(new_frontier->vertices + current_offset,
                           local_frontiers[i].data(),
                           num_nodes_in_local * sizeof(int));
                    current_offset += num_nodes_in_local;
                }
            }
            new_frontier->count = current_offset;

            // (c) Decision: Switch to Bottom-Up next iteration?
            //     If the new frontier is too large, switch
            if (new_frontier->count > threshold_td_to_bu)
            {
                use_top_down = false; // Switch to Bottom-Up
            }
        }
        else
        {
            // (a) Currently in Bottom-Up mode

            // Call your Task 3 (the one using schedule(dynamic, 768))
            bottom_up_step(graph, &local_frontiers, sol->distances, current_distance);

            // (b) Sequential merge (same as Bottom-Up)
            vertex_set_clear(new_frontier);
            int current_offset = 0;
            for (int i = 0; i < max_threads; ++i)
            {
                int num_nodes_in_local = local_frontiers[i].size();
                if (num_nodes_in_local > 0)
                {
                    memcpy(new_frontier->vertices + current_offset,
                           local_frontiers[i].data(),
                           num_nodes_in_local * sizeof(int));
                    current_offset += num_nodes_in_local;
                }
            }
            new_frontier->count = current_offset;
            
            // (c) Decision: Switch back to Top-Down next iteration?
            //     If the new frontier has become small again, switch back
            if (new_frontier->count < threshold_bu_to_td)
            {
                use_top_down = true; // Switch back to Top-Down
            }
        }


#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d (%s) %.4f sec\n", 
               frontier->count, 
               (use_top_down ? "TD" : "BU"), 
               end_time - start_time);
#endif

        // 10. Swap pointers (same as Top-Down / Bottom-Up)
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        // 11. Increment to the next level (same as Bottom-Up)
        current_distance++;
    }

    // 12. Cleanup (same as Top-Down / Bottom-Up)
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}
