import type { GraphData } from './types';

/**
 * Compute strongly connected components using Tarjan's algorithm
 */
function tarjanSCC(
  nodes: string[],
  adjacency: Map<string, string[]>
): string[][] {
  const index = new Map<string, number>();
  const lowlink = new Map<string, number>();
  const onStack = new Set<string>();
  const stack: string[] = [];
  const sccs: string[][] = [];
  let currentIndex = 0;

  function strongConnect(v: string) {
    index.set(v, currentIndex);
    lowlink.set(v, currentIndex);
    currentIndex++;
    stack.push(v);
    onStack.add(v);

    const neighbors = adjacency.get(v) || [];
    for (const w of neighbors) {
      if (!index.has(w)) {
        strongConnect(w);
        lowlink.set(v, Math.min(lowlink.get(v)!, lowlink.get(w)!));
      } else if (onStack.has(w)) {
        lowlink.set(v, Math.min(lowlink.get(v)!, index.get(w)!));
      }
    }

    if (lowlink.get(v) === index.get(v)) {
      const scc: string[] = [];
      let w: string;
      do {
        w = stack.pop()!;
        onStack.delete(w);
        scc.push(w);
      } while (w !== v);
      sccs.push(scc);
    }
  }

  for (const node of nodes) {
    if (!index.has(node)) {
      strongConnect(node);
    }
  }

  return sccs;
}

/**
 * Compute absorbing sets from graph data.
 * An absorbing set is a strongly connected component with no outgoing edges
 * to nodes outside the component.
 * 
 * Returns a map from node ID to absorbing set ID (or null if not in an absorbing set).
 */
export function computeAbsorbingSets(graphData: GraphData): Map<string, number | null> {
  const nodeIds = graphData.nodes.map(n => n.id);
  
  // Build adjacency list
  const adjacency = new Map<string, string[]>();
  for (const node of nodeIds) {
    adjacency.set(node, []);
  }
  for (const edge of graphData.edges) {
    const targets = adjacency.get(edge.source) || [];
    // Only add if not already present (avoid duplicates)
    if (!targets.includes(edge.target)) {
      targets.push(edge.target);
    }
    adjacency.set(edge.source, targets);
  }

  // Find all SCCs
  const sccs = tarjanSCC(nodeIds, adjacency);

  // Identify which SCCs are absorbing (no outgoing edges)
  const absorbingSCCs: string[][] = [];
  for (const scc of sccs) {
    const sccSet = new Set(scc);
    let hasOutgoingEdge = false;

    // Check if any node in the SCC has an edge to outside
    for (const node of scc) {
      const neighbors = adjacency.get(node) || [];
      for (const neighbor of neighbors) {
        if (!sccSet.has(neighbor)) {
          hasOutgoingEdge = true;
          break;
        }
      }
      if (hasOutgoingEdge) break;
    }

    // An absorbing set must have size > 1 or be a self-loop
    // (single nodes with no edges are not absorbing states in typical definition)
    if (!hasOutgoingEdge) {
      // Check if it's a valid absorbing set:
      // - Either has more than one node (strongly connected)
      // - Or has a self-loop
      if (scc.length > 1) {
        absorbingSCCs.push(scc);
      } else {
        const node = scc[0];
        const neighbors = adjacency.get(node) || [];
        if (neighbors.includes(node)) {
          // Has self-loop
          absorbingSCCs.push(scc);
        }
      }
    }
  }

  // Build result map
  const result = new Map<string, number | null>();
  for (const node of nodeIds) {
    result.set(node, null);
  }

  absorbingSCCs.forEach((scc, idx) => {
    for (const node of scc) {
      result.set(node, idx);
    }
  });

  return result;
}
