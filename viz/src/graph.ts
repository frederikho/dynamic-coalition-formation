import cytoscape, { Core, NodeSingular } from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import type { GraphData } from './types';

// Register layout algorithm
cytoscape.use(coseBilkent);

// Preset positions for 5-state case (3-country model)
// Pentagon layout matching the paper's figure
// Exact layout: ( ) top, (TC) upper-right, (WT) lower-right, (WTC) lower-left, (WC) upper-left
function getPentagonPositions(stateNames: string[]): Record<string, { x: number; y: number }> {
  if (stateNames.length !== 5) return {};
  
  // Pentagon coordinates - hardcoded for n=3
  const positions: Record<string, { x: number; y: number }> = {
    '( )': { x: 0, y: -180 },        // Top center
    '(TC)': { x: 171, y: -55 },      // Upper right
    '(WT)': { x: 106, y: 145 },      // Lower right
    '(WTC)': { x: -106, y: 145 },    // Lower left
    '(WC)': { x: -171, y: -55 },     // Upper left
    
    // Also support alternative orderings (CT, TW, CTW, CW, etc.)
    '(CT)': { x: 171, y: -55 },      // Upper right
    '(TW)': { x: 106, y: 145 },      // Lower right
    '(CTW)': { x: -106, y: 145 },    // Lower left
    '(TWC)': { x: -106, y: 145 },    // Lower left
    '(WCT)': { x: -106, y: 145 },    // Lower left
    '(CW)': { x: -171, y: -55 },     // Upper left
  };
  
  // Check if all states are in our predefined positions
  const allStatesKnown = stateNames.every(s => s in positions);
  if (!allStatesKnown) return {};
  
  const positionMap: Record<string, { x: number; y: number }> = {};
  stateNames.forEach(state => {
    positionMap[state] = positions[state];
  });
  
  return positionMap;
}

// Two-circle layout for n=4 case (15 states)
// Inner circle with 5 states, outer circle with 10 states
function getTwoCirclePositions(stateNames: string[]): Record<string, { x: number; y: number }> {
  if (stateNames.length !== 15) return {};
  const positions: Record<string, { x: number; y: number }> = {};

  // Inner preferred order (n=3 pentagon equivalents) - keep same layout
  const innerPreferred = ['( )', '(CT)', '(TW)', '(CTW)', '(CW)'];

  // Pick inner states as those not containing 'F' and matching preferred order first
  const innerStates: string[] = innerPreferred.filter(s => stateNames.includes(s));

  // If fewer than 5, add other states that do not include 'F'
  const nonFStates = stateNames.filter(s => !s.includes('F') && !innerStates.includes(s));
  for (const s of nonFStates) {
    if (innerStates.length >= 5) break;
    innerStates.push(s);
  }

  // If still fewer than 5, fill with any remaining states (shouldn't usually happen)
  for (const s of stateNames) {
    if (innerStates.length >= 5) break;
    if (!innerStates.includes(s)) innerStates.push(s);
  }

  const outerStates = stateNames.filter(s => !innerStates.includes(s)).sort();

  // Try to reuse pentagon positions for inner states (keeps identical layout to n=3)
  const pent = getPentagonPositions(innerStates);
  if (Object.keys(pent).length === innerStates.length) {
    // Use pentagon coordinates
    innerStates.forEach(state => { positions[state] = pent[state]; });
  } else {
    // Fallback: evenly spaced inner circle
    const innerRadius = 150;
    innerStates.forEach((state, i) => {
      const angle = (i / innerStates.length) * 2 * Math.PI - Math.PI / 2;
      positions[state] = { x: Math.cos(angle) * innerRadius, y: Math.sin(angle) * innerRadius };
    });
  }

  // Place outer circle states
  const outerRadius = 300;
  outerStates.forEach((state, i) => {
    const angle = (i / outerStates.length) * 2 * Math.PI - Math.PI / 2;
    positions[state] = { x: Math.cos(angle) * outerRadius, y: Math.sin(angle) * outerRadius };
  });

  return positions;
}

export class GraphRenderer {
  private cy: Core | null = null;
  private container: HTMLElement;
  private selectedNode: string | null = null;
  private onNodeSelect: ((nodeId: string | null) => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  setOnNodeSelect(callback: (nodeId: string | null) => void) {
    this.onNodeSelect = callback;
  }

  render(graphData: GraphData, probThreshold: number = 0.001) {
    // Destroy existing instance
    if (this.cy) {
      this.cy.destroy();
      this.cy = null;
    }

    // Filter edges by probability threshold
    const filteredEdges = graphData.edges.filter(e => e.p >= probThreshold);

    // Get preset positions for specific cases
    let presetPositions: Record<string, { x: number; y: number }> = {};
    if (graphData.nodes.length === 5) {
      presetPositions = getPentagonPositions(graphData.nodes.map(n => n.id));
    } else if (graphData.nodes.length === 15) {
      presetPositions = getTwoCirclePositions(graphData.nodes.map(n => n.id));
    }
    const usePresetPositions = Object.keys(presetPositions).length > 0;

    // Convert to Cytoscape format
    const elements: cytoscape.ElementDefinition[] = [
      // Nodes
      ...graphData.nodes.map(node => {
        const element: cytoscape.ElementDefinition = {
          group: 'nodes' as const,
          data: {
            id: node.id,
            label: node.label,
            geo_level: node.meta?.geo_level || 0
          }
        };

        // Add preset position if available
        if (usePresetPositions && node.id in presetPositions) {
          element.position = presetPositions[node.id];
        }

        return element;
      }),
      // Edges
      ...filteredEdges.map(edge => ({
        group: 'edges' as const,
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.p.toFixed(3),
          probability: edge.p,
          isSelfLoop: edge.source === edge.target,
          width: this.getEdgeWidth(edge.p)
        }
      }))
    ];

    // Create Cytoscape instance
    this.cy = cytoscape({
      container: this.container,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#cbd5e1',
            'label': 'data(label)',
            'color': '#1e293b',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '14px',
            'font-weight': 'normal',
            'width': 60,
            'height': 60,
            'text-outline-width': 0
          }
        },
        {
          selector: 'node:selected',
          style: {
            'background-color': '#94a3b8',
            'border-width': 3,
            'border-color': '#475569'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(width)',
            'line-color': '#334155',
            'target-arrow-color': '#334155',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '11px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
            'color': '#1e293b',
            'text-background-color': 'transparent',
            'text-background-opacity': 0
          }
        },
        {
          selector: 'edge[isSelfLoop]',
          style: {
            'curve-style': 'loop',
            'loop-direction': '0deg',
            'loop-sweep': '90deg',
            'control-point-step-size': 40
          }
        },
        {
          selector: '.highlighted',
          style: {
            'line-color': '#0f172a',
            'target-arrow-color': '#0f172a',
            'opacity': 1
          }
        },
        {
          selector: '.dimmed',
          style: {
            'opacity': 0.2
          }
        }
      ],
      layout: usePresetPositions ? {
        name: 'preset'
      } : {
        name: 'cose-bilkent',
        animate: false,
        randomize: true,
        nodeRepulsion: 4500,
        idealEdgeLength: 100,
        edgeElasticity: 0.45,
        nestingFactor: 0.1,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
        tilingPaddingVertical: 10,
        tilingPaddingHorizontal: 10
      },
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.2
    });

    this.setupInteractions();
    
    // For larger graphs (n=4 with 15 nodes), fit to view with padding
    if (graphData.nodes.length >= 15) {
      this.cy.fit(undefined, 50); // Fit all elements with 50px padding
    }
  }

  private getEdgeWidth(probability: number): number {
    // Width scales with probability: min 1px, max 8px
    // Linear scaling: width = 1 + 7 * probability
    return Math.max(1, Math.min(8, 1 + 7 * probability));
  }

  private setupInteractions() {
    if (!this.cy) return;

    // Click node to select
    this.cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      if (this.selectedNode === node.id()) {
        this.deselectNode();
      } else {
        this.selectNode(node.id());
      }
    });

    // Click background to deselect
    this.cy.on('tap', (evt) => {
      if (evt.target === this.cy) {
        this.deselectNode();
      }
    });

    // Hover effects
    this.cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;

      // Highlight connected edges
      const connectedEdges = node.connectedEdges();
      connectedEdges.addClass('highlighted');

      // Dim everything else
      this.cy?.elements().not(node).not(connectedEdges).addClass('dimmed');
    });

    this.cy.on('mouseout', 'node', () => {
      this.cy?.elements().removeClass('highlighted dimmed');
    });
  }

  private selectNode(nodeId: string) {
    if (!this.cy) return;

    this.selectedNode = nodeId;

    // Deselect all and select the clicked node
    this.cy.nodes().unselect();
    const node = this.cy.$id(nodeId);
    node.select();

    if (this.onNodeSelect) {
      this.onNodeSelect(nodeId);
    }
  }

  private deselectNode() {
    if (!this.cy) return;

    this.selectedNode = null;
    this.cy.nodes().unselect();

    if (this.onNodeSelect) {
      this.onNodeSelect(null);
    }
  }

  resetView() {
    if (!this.cy) return;
    this.cy.fit(undefined, 50);
  }

  getNodeData(nodeId: string): any {
    if (!this.cy) return undefined;
    const node = this.cy.$id(nodeId);
    if (node.length === 0) return undefined;
    return {
      label: node.data('label'),
      meta: {
        geo_level: node.data('geo_level')
      }
    };
  }

  getOutgoingEdges(nodeId: string): Array<{ target: string; probability: number }> {
    if (!this.cy) return [];

    const node = this.cy.$id(nodeId);
    if (node.length === 0) return [];

    const edges: Array<{ target: string; probability: number }> = [];
    node.outgoers('edge').forEach(edge => {
      edges.push({
        target: edge.target().id(),
        probability: edge.data('probability')
      });
    });

    return edges.sort((a, b) => b.probability - a.probability);
  }

  getIncomingEdges(nodeId: string): Array<{ source: string; probability: number }> {
    if (!this.cy) return [];

    const node = this.cy.$id(nodeId);
    if (node.length === 0) return [];

    const edges: Array<{ source: string; probability: number }> = [];
    node.incomers('edge').forEach(edge => {
      edges.push({
        source: edge.source().id(),
        probability: edge.data('probability')
      });
    });

    return edges.sort((a, b) => b.probability - a.probability);
  }

  destroy() {
    if (this.cy) {
      this.cy.destroy();
      this.cy = null;
    }
  }
}
