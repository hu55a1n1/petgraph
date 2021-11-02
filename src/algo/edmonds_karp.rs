use std::fmt::Debug;
use std::hash::Hash;
use std::collections::{VecDeque, HashMap, HashSet};
use std::cmp;
use std::cmp::Ord;
use std::ops::{Sub, Add};
use num::Zero;
use crate::graph::{NodeIndex, DiGraph, IndexType, EdgeIndex};
use crate::Graph;
use crate::visit::{VisitMap, GraphRef, Visitable, IntoNeighbors, IntoNodeReferences};
use crate::visit::{NodeCount, NodeIndexable, NodeRef, GraphBase, IntoEdges, EdgeRef};

/// \[Generic\] [Edmonds-Karp algorithm](https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm)
///
/// Computes the max flow in the graph.
/// Edge weights are assumed to be nonnegative.
/// 
/// # Arguments
/// * `graph`: graph with nonnegative edge weights.
/// * `start`: graph node where the flow starts.
/// * `end`: graph node where the flow ends.
///
/// # Returns
/// * Max flow from `start` to `end`.
/// 
/// Running time is O(|V||E|^2), where |V| is the number of vertices and |E| is the number of edges.
/// Dinic's algorithm solves this problem in O(|V|^2|E|).
/// TODO: Do not remove edges from the graph

pub fn edmonds_karp<V, E>(
    original_graph: &Graph<V, E>, 
    start: NodeIndex, 
    end: NodeIndex
) -> E
where
    V: Clone + Debug,
    E: Zero + Ord + Copy + Sub<Output = E> + Add<Output = E> + Debug,
{
    // Start by making a directed version of the original graph using BFS.
    // The graph must be copied as an adjacency list in order to run BFS in O(|E|) time.
    let mut graph = copy_graph_directed(original_graph).unwrap();
    let mut max_flow = E::zero();
    
    // This loop will run O(|V||E|) times. Each iteration takes O(|E|) time.
    loop {
        let mut second = end;
        let path = BfsPath::shortest_path(&graph, start, end);
        if path.len() == 1 {
            break;
        }

        let path_flow = min_weight(&graph, path.clone());
        println!("path {:?} flow {:?}", path, path_flow);
        max_flow = max_flow + path_flow;

        for node in path.into_iter().rev().skip(1) {
            let first = node;
            let edge = graph.find_edge(first, second).expect("Edge should be in graph");
            let weight = &mut graph[edge];
            if *weight == path_flow {
                graph.remove_edge(edge);
            } else {
                *weight = *weight - path_flow;
            }

            // Add reverse edge to make the residual graph.
            match graph.find_edge(second, first) {
                None => {
                    graph.add_edge(second, first, path_flow);
                }
                Some(edge) => {
                    graph.update_edge(second, first, path_flow + graph[edge]);
                }
            }
            second = first;
        }
    }
    max_flow
}

// Finds the minimum edge weight along the path.
fn min_weight<V, E>(graph: &Graph<V, E>, path: Vec<EdgeIndex>) -> E 
where
    E: Zero + Ord + Copy,
{
    let mut iter = path.into_iter();
    if let Some(edge) = iter.next() {
        let mut weight = graph.edge_weight(edge).expect("Edge should be in graph.");
        let mut first = second;
        for second in iter {
            if let Some(edge) = graph.find_edge(first, second) {
                weight = cmp::min(weight, graph.edge_weight(edge).expect("Edge should be in graph."));
                first = second;
            } else {
                return E::zero();
            }
        }
        return *weight;
    }
    E::zero()
}

/// Creates a copy of original_graph and stores it as a directed adjacency list.
/// If n -> n' is an edge, it also adds the edge n' -> n but with weight 0.
pub fn copy_graph_directed<G, V, E, N, I, ER>(original_graph: G) -> Result<DiGraph<V, E>, String>
where
    G: GraphBase<NodeId = I> + IntoEdges<EdgeRef = ER> + IntoNodeReferences<NodeRef = N>,
    N: NodeRef<NodeId = I, Weight = V>,
    ER: EdgeRef<NodeId = I, Weight = E>,
    I: Hash + Eq,
    E: Clone + Zero + PartialOrd,
    V: Clone,
{
    let mut graph_copy: DiGraph<V, E> = Graph::default();
    // Ids of new nodes
    let mut new_node_ids = Vec::new();
    // All nodes in the graph
    let node_references: Vec<_> = original_graph.node_references().collect();

    // Add all nodes into graph_copy and keep track of their new index
    for node in node_references.iter() {
        let id = graph_copy.add_node(node.weight().clone());
        new_node_ids.push(id);
    }

    // Store the index of a node in the vector node_references
    let index_map: HashMap<_, _> = node_references
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id(), index))
        .collect();
    
    // Extra edges to add to graph_copy
    let mut extra_edges = HashSet::new();
    
    for start_ref in node_references {
        let edges = original_graph.edges(start_ref.id());
        for edge_ref in edges {
            let start_index = index_map[&start_ref.id()];
            let end_index = index_map[&edge_ref.target()];
            
            // We don't need to add the reversed edge
            if start_index > end_index {
                extra_edges.remove(&(end_index, start_index));
            // We need to add the reversed edge
            } else {
                extra_edges.insert((end_index, start_index));
            }
            let weight = edge_ref.weight().clone();
            if weight < E::zero() {
                return Err("Nonnegative edgeweights expected for Edmonds-Karp.".to_owned());
            }
            graph_copy.add_edge(new_node_ids[start_index], new_node_ids[end_index], weight);
        }
    }

    for (index1, index2) in extra_edges {
        graph_copy.add_edge(new_node_ids[index1], new_node_ids[index2], E::zero());
    }
    Ok(graph_copy)
}

/// Same as crate::visit::Bfs but uses Bfs to compute the shortest path in an unweighted graph.
#[derive(Clone)]
pub struct BfsPath<N, VM> {
    /// The queue of nodes to visit
    pub stack: VecDeque<N>,
    /// The map of discovered nodes
    pub discovered: VM, 
}

impl<N, VM> Default for BfsPath<N, VM>
where
    VM: Default,
{
    fn default() -> Self {
        BfsPath {
            stack: VecDeque::new(),
            discovered: VM::default(),
        }
    }
}

impl<N, VM> BfsPath<N, VM>
where
    N: Copy + PartialEq,
    VM: VisitMap<N>,
{
    /// Create a new **Bfs**, using the graph's visitor map, and put **start**
    /// in the stack of nodes to visit.
    fn new<G>(graph: G, start: N) -> Self
    where
        G: GraphRef + Visitable<NodeId = N, Map = VM>,
    {
        let mut discovered = graph.visit_map();
        discovered.visit(start);
        let mut stack = VecDeque::new();
        stack.push_front(start);
        BfsPath { stack, discovered }
    }

    /// Returns a shortest path from start to end ignoring edge weights.
    /// The path is a vector of EdgeRef.
    pub fn shortest_path<G, ER>(graph: G, start: N, end: N) -> Vec<<ER as EdgeRef>::EdgeId>
    where
        G: GraphRef + Visitable<NodeId = N, Map = VM> + NodeCount,
        G: IntoEdges<EdgeRef = ER> + NodeIndexable,
        ER: EdgeRef<NodeId = N>,
        N: Debug,
    {
        // For every Node N in G, stores the EdgeRef that first goes to N
        let mut predecessor: Vec<Option<_>> = vec![None; graph.node_count()];
        let mut path: Vec<<ER as EdgeRef>::EdgeId> = Vec::new();
        let mut bfs = BfsPath::new(graph, start);

        while let Some(node) = bfs.stack.pop_front() {
            if node == end {
                break;
            }
            for edge in graph.edges(node) {
                let succ = edge.target();
                if bfs.discovered.visit(succ) {
                    bfs.stack.push_back(succ);
                    predecessor[graph.to_index(succ)] = Some(edge);
                }
            }
        } 

        let mut next = end;
        while let Some(edge) = predecessor[graph.to_index(next)] {
            path.push(edge.id());
            let node = edge.source();
            if node == start {
                break;
            }
            next = node;
        }
        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_flow_unweighted() {
        let mut graph = Graph::<_, u32>::new();
        let v0 = graph.add_node(0);
        let v1 = graph.add_node(1);
        let v2 = graph.add_node(2);
        let v3 = graph.add_node(3);
        let v4 = graph.add_node(4);

        graph.extend_with_edges(&[
            (v0, v1, 1), (v0, v2, 1),
            (v2, v3, 1), (v3, v4, 1), (v2, v4, 1),
        ]);
        // 0 ---> 1
        // |      
        // v
        // 2 ---> 4
        // |     7
        // v   /
        // 3
        assert_eq!(1, edmonds_karp(&graph, v0, v4));

        graph.add_edge(v1, v4, 1);
        assert_eq!(2, edmonds_karp(&graph, v0, v4));

        graph.add_edge(v0, v3, 1);
        assert_eq!(3, edmonds_karp(&graph, v0, v4));

        graph.clear();
        graph.extend_with_edges(&[
            (v0, v1, 1), (v0, v2, 1), (v1, v4, 1),
            (v2, v3, 1), (v4, v3, 1), (v2, v4, 1),
        ]);
        assert_eq!(2, edmonds_karp(&graph, v0, v4));
    }

    #[test]
    fn test_min_weight() {
        let mut graph = Graph::<i32, u32>::new();
        let v0 = graph.add_node(0);
        let v1 = graph.add_node(0);
        let v2 = graph.add_node(0);
        let v3 = graph.add_node(0);
        graph.extend_with_edges(&[
            (v0, v1, 3), (v1, v2, 3),
            (v2, v3, 4)
        ]);
        let path = vec![v0, v1, v2, v3];
        assert_eq!(3, min_weight(&graph, path));
        let path = vec![v0, v2, v1, v3];
        assert_eq!(0, min_weight(&graph, path));
    }

    #[test]
    fn test_max_flow_weighted() {
        let mut graph = Graph::<_, u32>::new();
        let v0 = graph.add_node(0);
        let v1 = graph.add_node(1);
        let v2 = graph.add_node(2);
        let v3 = graph.add_node(3);
        graph.extend_with_edges(&[
            (v1, v2, 3), (v1, v3, 1), (v2, v3, 3),
            (v2, v0, 1), (v3, v0, 3)
        ]);
        let max_flow = edmonds_karp(&graph, v1, v0);
        assert_eq!(4, max_flow);

        let mut graph = Graph::<_, u32>::new();
        let a1 = graph.add_node(0);
        let b1 = graph.add_node(0);
        let b2 = graph.add_node(0);
        let b3 = graph.add_node(0);
        let c1 = graph.add_node(0);
        let c2 = graph.add_node(0);
        let c3 = graph.add_node(0);
        let d1 = graph.add_node(0);
        graph.extend_with_edges(&[
            (a1, b1, 6), (a1, b2, 1), (a1, b3, 1),
            (b1, c1, 6), (b1, c2, 6),
            (b2, c1, 1), (b2, c3, 1),
            (b3, c2, 1), (b3, c3, 1),
            (c1, d1, 1), (c2, d1, 4), (c3, d1, 3)
        ]);
        let max_flow = edmonds_karp(&graph, a1, d1);
        assert_eq!(7, max_flow);

        let mut graph = Graph::<_, u32>::new();
        let a1 = graph.add_node(0);
        let b1 = graph.add_node(0);
        let b2 = graph.add_node(0);
        let b3 = graph.add_node(0);
        let c1 = graph.add_node(0);
        let c2 = graph.add_node(0);
        let d1 = graph.add_node(0);
        graph.extend_with_edges(&[
            (a1, b1, 20), (a1, b2, 40), (a1, b3, 5),
            (b1, b2, 5), (b2, b3, 5),
            (b1, c1, 20), (b2, c1, 25), (b2, c2, 15), (b3, c2, 10),
            (c1, c2, 5),
            (c1, d1, 40), (c2, d1, 30),
        ]);
        let max_flow = edmonds_karp(&graph, a1, d1);
        assert_eq!(65, max_flow);
    }
}