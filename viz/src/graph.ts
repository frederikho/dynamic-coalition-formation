import Graph from 'graphology';
import Sigma from 'sigma';
import { Attributes } from 'graphology-types';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import circular from 'graphology-layout/circular';
import type { GraphData, GraphEdge } from './types';

export class GraphRenderer {
  private graph: Graph;
  private sigma: Sigma | null = null;
  private container: HTMLElement;
  private selectedNode: string | null = null;
  private onNodeSelect: ((nodeId: string | null) => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.graph = new Graph({ multi: true, type: 'directed' });
  }

  setOnNodeSelect(callback: (nodeId: string | null) => void) {
    this.onNodeSelect = callback;
  }

  render(graphData: GraphData, probThreshold: number = 0.001) {
    // Clear existing graph
    this.graph.clear();
    if (this.sigma) {
      this.sigma.kill();
      this.sigma = null;
    }

    // Filter edges by probability threshold
    const filteredEdges = graphData.edges.filter(e => e.p >= probThreshold);

    // Add nodes
    graphData.nodes.forEach(node => {
      this.graph.addNode(node.id, {
        label: node.label,
        x: node.x ?? 0,
        y: node.y ?? 0,
        size: 15,
        color: '#2563eb',
        meta: node.meta
      });
    });

    // Add edges (including self-loops)
    filteredEdges.forEach(edge => {
      this.graph.addEdge(edge.source, edge.target, {
        type: 'arrow',
        label: edge.p.toFixed(3),
        size: 2,
        color: this.getEdgeColor(edge.p),
        probability: edge.p,
        isSelfLoop: edge.source === edge.target,
        // Self-loops will be rendered with default arrow type
        // Sigma.js handles them reasonably well
      });
    });

    // Apply layout if no positions provided
    if (!graphData.nodes[0]?.x) {
      this.applyLayout();
    }

    // Create Sigma instance
    this.sigma = new Sigma(this.graph, this.container, {
      renderEdgeLabels: true,
      defaultEdgeType: 'arrow',
      labelSize: 12,
      labelWeight: 'bold',
      edgeLabelSize: 10,
      edgeLabelColor: { color: '#666' },
    });

    // Setup interactions
    this.setupInteractions();
  }

  private applyLayout() {
    // First apply circular layout for initial positions
    circular.assign(this.graph);

    // Then refine with force atlas
    const settings = forceAtlas2.inferSettings(this.graph);
    forceAtlas2.assign(this.graph, {
      iterations: 500,
      settings: {
        ...settings,
        scalingRatio: 10,
        gravity: 1,
        barnesHutOptimize: false
      }
    });
  }

  private getEdgeColor(probability: number): string {
    // Color based on probability: red (low) -> yellow (mid) -> green (high)
    if (probability >= 0.5) return '#16a34a'; // green
    if (probability >= 0.1) return '#eab308'; // yellow
    return '#94a3b8'; // gray
  }

  private setupInteractions() {
    if (!this.sigma) return;

    // Click to select node
    this.sigma.on('clickNode', ({ node }) => {
      if (this.selectedNode === node) {
        this.deselectNode();
      } else {
        this.selectNode(node);
      }
    });

    // Click stage to deselect
    this.sigma.on('clickStage', () => {
      this.deselectNode();
    });

    // Hover effects
    let hoveredNode: string | null = null;
    let hoveredNeighbors: Set<string> | null = null;

    this.sigma.on('enterNode', ({ node }) => {
      hoveredNode = node;
      hoveredNeighbors = new Set(this.graph.neighbors(node));
      hoveredNeighbors.add(node);

      this.sigma?.setSetting('nodeReducer', (n, data) => {
        if (hoveredNeighbors?.has(n)) {
          return { ...data };
        }
        return { ...data, color: '#ddd', hidden: false };
      });

      this.sigma?.setSetting('edgeReducer', (edge, data) => {
        const source = this.graph.source(edge);
        const target = this.graph.target(edge);
        if (source === node || target === node) {
          return { ...data };
        }
        return { ...data, color: '#e5e5e5', hidden: false };
      });

      this.sigma?.refresh();
    });

    this.sigma.on('leaveNode', () => {
      hoveredNode = null;
      hoveredNeighbors = null;

      this.sigma?.setSetting('nodeReducer', null);
      this.sigma?.setSetting('edgeReducer', null);
      this.sigma?.refresh();
    });
  }

  private selectNode(nodeId: string) {
    this.selectedNode = nodeId;

    // Highlight selected node
    this.sigma?.setSetting('nodeReducer', (node, data) => {
      if (node === nodeId) {
        return { ...data, color: '#dc2626', highlighted: true };
      }
      return { ...data };
    });

    this.sigma?.refresh();

    if (this.onNodeSelect) {
      this.onNodeSelect(nodeId);
    }
  }

  private deselectNode() {
    this.selectedNode = null;

    this.sigma?.setSetting('nodeReducer', null);
    this.sigma?.refresh();

    if (this.onNodeSelect) {
      this.onNodeSelect(null);
    }
  }

  resetView() {
    this.sigma?.getCamera().setState({ x: 0.5, y: 0.5, angle: 0, ratio: 1 });
  }

  getNodeData(nodeId: string): Attributes | undefined {
    if (!this.graph.hasNode(nodeId)) return undefined;
    return this.graph.getNodeAttributes(nodeId);
  }

  getOutgoingEdges(nodeId: string): Array<{ target: string; probability: number }> {
    if (!this.graph.hasNode(nodeId)) return [];

    const edges: Array<{ target: string; probability: number }> = [];
    this.graph.forEachOutEdge(nodeId, (edge, attrs, source, target) => {
      edges.push({
        target,
        probability: attrs.probability
      });
    });

    return edges.sort((a, b) => b.probability - a.probability);
  }

  getIncomingEdges(nodeId: string): Array<{ source: string; probability: number }> {
    if (!this.graph.hasNode(nodeId)) return [];

    const edges: Array<{ source: string; probability: number }> = [];
    this.graph.forEachInEdge(nodeId, (edge, attrs, source, target) => {
      edges.push({
        source,
        probability: attrs.probability
      });
    });

    return edges.sort((a, b) => b.probability - a.probability);
  }

  destroy() {
    if (this.sigma) {
      this.sigma.kill();
      this.sigma = null;
    }
    this.graph.clear();
  }
}
