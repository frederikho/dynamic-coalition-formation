import cytoscape, { Core, NodeSingular } from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import cola from 'cytoscape-cola';
import type { GraphData } from './types';
import { computeAbsorbingSets } from './absorbing';

// Register layout algorithms
cytoscape.use(coseBilkent);
cytoscape.use(cola);

// Standard player order for normalization
const STANDARD_ORDER = ['H', 'W', 'T', 'C', 'F', 'A', 'B', 'D', 'E', 'G'];

function normalizeStateName(stateName: string): string {
  /**
   * Normalize state name to use standard player order (H, W, T, C, F, ...).
   * E.g., "(CTW)" -> "(WTC)", "(CF)(TW)" -> "(CF)(TW)"
   */
  if (stateName === '( )') return stateName;

  // Extract all coalitions from the state name
  const coalitionMatches = stateName.match(/\([A-Z]+\)/g);
  if (!coalitionMatches) return stateName;

  // Normalize each coalition
  const normalizedCoalitions = coalitionMatches.map(coalition => {
    const members = coalition.slice(1, -1).split(''); // Remove parens and split
    const sorted = members.sort((a, b) => {
      const indexA = STANDARD_ORDER.indexOf(a);
      const indexB = STANDARD_ORDER.indexOf(b);
      return (indexA === -1 ? 999 : indexA) - (indexB === -1 ? 999 : indexB);
    });
    return `(${sorted.join('')})`;
  });

  return normalizedCoalitions.join('');
}

// Export for use in other modules
(window as any).normalizeStateName = normalizeStateName;

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
  private savedPositions: Record<string, { x: number; y: number }> = {};

  constructor(container: HTMLElement) {
    this.container = container;
  }

  setOnNodeSelect(callback: (nodeId: string | null) => void) {
    this.onNodeSelect = callback;
  }

  private saveCurrentPositions() {
    if (!this.cy) return;
    this.savedPositions = {};
    this.cy.nodes().forEach(node => {
      const pos = node.position();
      this.savedPositions[node.id()] = { x: pos.x, y: pos.y };
    });
  }

  private computeClusterPositions(
    layoutMode: 'default' | 'connections' | 'deployer' | 'geo-level',
    graphData: GraphData
  ): Record<string, { x: number; y: number }> {
    if (layoutMode !== 'deployer' && layoutMode !== 'geo-level') {
      return {};
    }

    // Group nodes by cluster key (deployer or G level)
    const nodesByCluster = new Map<string, string[]>();
    graphData.nodes.forEach(node => {
      let clusterKey: string;
      if (layoutMode === 'deployer') {
        clusterKey = node.meta?.deploying_coalition || 'None';
      } else {
        const gLevel = node.meta?.geo_level || 0;
        clusterKey = Math.round(gLevel).toString();
      }

      if (!nodesByCluster.has(clusterKey)) {
        nodesByCluster.set(clusterKey, []);
      }
      nodesByCluster.get(clusterKey)!.push(node.id);
    });

    // Position cluster centers in a large circle
    const clusters = Array.from(nodesByCluster.keys());
    const numClusters = clusters.length;
    const clusterRadius = 280; // Large radius for separation

    const clusterCenters = new Map<string, {x: number, y: number}>();
    clusters.forEach((cluster, i) => {
      const angle = (i / numClusters) * 2 * Math.PI - Math.PI / 2;
      clusterCenters.set(cluster, {
        x: Math.cos(angle) * clusterRadius,
        y: Math.sin(angle) * clusterRadius
      });
    });

    // Pre-position nodes near their cluster centers
    const nodePositions: Record<string, {x: number, y: number}> = {};
    const nodeDiameter = 70; // Node size from styles

    nodesByCluster.forEach((nodeIds, cluster) => {
      const center = clusterCenters.get(cluster)!;
      const numNodes = nodeIds.length;

      // Calculate minimum radius to prevent overlap
      // Chord length between adjacent nodes must be > nodeDiameter
      // Chord = 2R * sin(π/N), so R = (nodeDiameter * margin) / (2 * sin(π/N))
      let nodeRadius: number;
      if (numNodes === 1) {
        nodeRadius = 0; // Single node at center
      } else {
        const margin = 1.3; // 30% spacing margin
        const minChordLength = nodeDiameter * margin;
        nodeRadius = minChordLength / (2 * Math.sin(Math.PI / numNodes));
      }

      nodeIds.forEach((nodeId, i) => {
        const angle = (i / numNodes) * 2 * Math.PI;
        nodePositions[nodeId] = {
          x: center.x + Math.cos(angle) * nodeRadius,
          y: center.y + Math.sin(angle) * nodeRadius
        };
      });
    });

    return nodePositions;
  }

  render(graphData: GraphData, probThreshold: number = 0.001, options?: {
    coloringMode?: 'none' | 'absorbing' | 'geoengineering' | 'deployer';
    filterMode?: 'absolute' | 'cumulative';
    layoutMode?: 'default' | 'connections' | 'deployer' | 'geo-level';
    // UI toggles
    showSelfLoops?: boolean;
    showEdgeLabels?: boolean;
    showNodeLabels?: boolean;
  }) {
    // Save current positions before destroying
    if (this.cy) {
      this.saveCurrentPositions();
      this.cy.destroy();
      this.cy = null;
    }

    const filterMode = options?.filterMode || 'absolute';
    const coloringMode = options?.coloringMode || 'none';
    const layoutMode = options?.layoutMode || 'default';
    const showSelfLoops = options?.showSelfLoops !== false; // default true
    const showEdgeLabels = options?.showEdgeLabels !== false; // default true
    const showNodeLabels = options?.showNodeLabels !== false; // default true

    // Filter edges by probability threshold
    let filteredEdges: typeof graphData.edges;

    if (filterMode === 'cumulative') {
      // Cumulative tail filtering: for each target node, remove smallest incoming edges until sum reaches threshold
      const edgesToRemove = new Set<string>();

      // Group edges by target node
      const edgesByTarget = new Map<string, typeof graphData.edges>();
      for (const edge of graphData.edges) {
        if (!edgesByTarget.has(edge.target)) {
          edgesByTarget.set(edge.target, []);
        }
        edgesByTarget.get(edge.target)!.push(edge);
      }

      // For each target node, identify edges to remove
      for (const [target, edges] of edgesByTarget) {
        // Sort by probability ascending (smallest first)
        const sorted = [...edges].sort((a, b) => a.p - b.p);

        let cumSum = 0;
        for (const edge of sorted) {
          if (cumSum + edge.p <= probThreshold) {
            // Remove this edge
            edgesToRemove.add(edge.id);
            cumSum += edge.p;
          } else {
            // Stop - keep this and all remaining edges
            break;
          }
        }
      }

      filteredEdges = graphData.edges.filter(e => !edgesToRemove.has(e.id));
    } else {
      // Absolute threshold filtering
      filteredEdges = graphData.edges.filter(e => e.p >= probThreshold);
    }

    // Get preset positions for specific cases
    let presetPositions: Record<string, { x: number; y: number }> = {};
    if (graphData.nodes.length === 5) {
      presetPositions = getPentagonPositions(graphData.nodes.map(n => n.id));
    } else if (graphData.nodes.length === 15) {
      presetPositions = getTwoCirclePositions(graphData.nodes.map(n => n.id));
    }

    // Compute cluster positions for clustering layouts
    const clusterPositions = this.computeClusterPositions(layoutMode, graphData);

    // Position priority: saved > cluster/preset positions
    // This ensures that user interactions (dragging) or previous renders are preserved
    const finalPositions: Record<string, { x: number; y: number }> = {};

    for (const nodeId of graphData.nodes.map(n => n.id)) {
      if (this.savedPositions[nodeId]) {
        // Always prioritize saved positions from previous render
        finalPositions[nodeId] = this.savedPositions[nodeId];
      } else if (layoutMode === 'deployer' || layoutMode === 'geo-level') {
        // For clustering modes: use computed cluster positions
        if (clusterPositions[nodeId]) {
          finalPositions[nodeId] = clusterPositions[nodeId];
        }
      } else if (layoutMode === 'default' && presetPositions[nodeId]) {
        // Only apply preset positions for the 'default' layout
        finalPositions[nodeId] = presetPositions[nodeId];
      }
    }

    const usePresetPositions = Object.keys(finalPositions).length > 0;

    // Prepare coloring based on mode
    const palette = ['#e11d48','#06b6d4','#84cc16','#f59e0b','#7c3aed','#10b981','#0ea5e9','#f97316','#6366f1','#db2777','#14b8a6','#f43f5e'];

    // Compute absorbing sets if needed
    const nodeToAbsorbing = coloringMode === 'absorbing' ? computeAbsorbingSets(graphData, filteredEdges) : new Map<string, number | null>();
    const absorbingSetIds = new Set<number>();
    if (coloringMode === 'absorbing') {
      nodeToAbsorbing.forEach(setId => {
        if (setId !== null) absorbingSetIds.add(setId);
      });
    }

    // Compute G level range for geoengineering coloring
    let minG = 0, maxG = 1;
    if (coloringMode === 'geoengineering') {
      const geoLevels = graphData.nodes.map(n => n.meta?.geo_level || 0);
      minG = Math.min(...geoLevels);
      maxG = Math.max(...geoLevels);
    }

    // Get unique deployers for deployer coloring
    const deployerToColor = new Map<string, string>();
    if (coloringMode === 'deployer') {
      const uniqueDeployers = Array.from(new Set(
        graphData.nodes.map(n => n.meta?.deploying_coalition).filter(d => d !== undefined)
      )).sort() as string[];
      uniqueDeployers.forEach((deployer, i) => {
        deployerToColor.set(deployer, palette[i % palette.length]);
      });
    }

    // If requested, remove self-loop edges from display
    const displayedEdgesList = showSelfLoops ? filteredEdges : filteredEdges.filter(e => e.source !== e.target);

    // Convert to Cytoscape format
    const elements: cytoscape.ElementDefinition[] = [
      // Nodes
      ...graphData.nodes.map(node => {
        const geoLevel = node.meta?.geo_level || 0;
        const deployingCoalition = node.meta?.deploying_coalition || '';
        const normalizedLabel = normalizeStateName(node.label);
        const label = `${normalizedLabel}\nG=${geoLevel.toFixed(2)}`;

        // Border should only appear when the state has exactly one coalition (the deployer)
        // Count coalitions in state name: "(CF)" has 1, "(CF)(TW)" has 2, "( )" has 0
        const coalitionCount = (node.id.match(/\([A-Z]+\)/g) || []).length;

        // Border if: only when coloring by deployer, G > 0, has exactly one coalition, and it matches the deployer
        // Compare normalized versions to handle different orderings
        const normalizedState = normalizeStateName(node.id);
        const hasDeploymentBorder = coloringMode === 'deployer' && geoLevel > 0 && coalitionCount === 1 && normalizedState === deployingCoalition;

        const element: cytoscape.ElementDefinition = {
          group: 'nodes' as const,
          data: {
            id: node.id,
            label: label,
            geo_level: geoLevel,
            deploying_coalition: deployingCoalition,
            has_deployment: hasDeploymentBorder
          }
        };

        // Add preset position if available (prefers saved positions)
        if (usePresetPositions && node.id in finalPositions) {
          element.position = finalPositions[node.id];
        }

        // Assign color based on coloring mode
        let color = '#cbd5e1'; // Default gray

        if (coloringMode === 'absorbing') {
          const setId = nodeToAbsorbing.get(node.id) ?? null;
          if (setId !== null) {
            // Sort set IDs to get consistent coloring
            const sortedIds = Array.from(absorbingSetIds).sort((a, b) => a - b);
            const idx = sortedIds.indexOf(setId);
            color = palette[idx % palette.length];
          }
        } else if (coloringMode === 'geoengineering') {
          // Blue gradient based on G level
          const geoLevel = node.meta?.geo_level || 0;
          const range = maxG - minG || 1;
          const normalized = (geoLevel - minG) / range;

          // Interpolate from light blue (#e0f2fe) to dark blue (#0369a1)
          // Light blue for low G, dark blue for high G
          const r = Math.round(224 - normalized * (224 - 3));
          const g = Math.round(242 - normalized * (242 - 105));
          const b = Math.round(254 - normalized * (254 - 161));
          color = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
        } else if (coloringMode === 'deployer') {
          // Color by deploying coalition
          const deployer = node.meta?.deploying_coalition;
          if (deployer && deployerToColor.has(deployer)) {
            color = deployerToColor.get(deployer)!;
          }
        }

        (element.data as any).color = color;

        return element;
      }),
      // Edges
      ...displayedEdgesList.map(edge => ({
        group: 'edges' as const,
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          // keep probability data but label rendering can be toggled via styles
          label: this.formatProbability(edge.p),
          probability: edge.p,
          isSelfLoop: edge.source === edge.target,
          width: this.getEdgeWidth(edge.p, edge.source === edge.target, filteredEdges)
        }
      }))
    ];

    // Build style array dynamically based on toggles
    const styles: any[] = [];

    // Node base style (use text-opacity to reliably hide/show labels)
    styles.push({
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        'label': 'data(label)',
        'text-opacity': showNodeLabels ? 1 : 0,
        'color': '#1e293b',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': '12px',
        'font-weight': 'normal',
        'width': 70,
        'height': 70,
        'text-outline-width': 0,
        'text-wrap': 'wrap',
        'text-max-width': '65px',
        'border-width': 0,
        'border-color': '#1e293b'
      }
    });

    // Deployment border
    styles.push({
      selector: 'node[has_deployment]',
      style: {
        'border-width': function(ele: any) {
          return ele.data('has_deployment') ? 3 : 0;
        },
        'border-color': '#1e293b'
      }
    });

    // Selected node
    styles.push({
      selector: 'node:selected',
      style: {
        'background-color': '#94a3b8',
        'border-width': 3,
        'border-color': '#475569'
      }
    });

    // Edge base style (use text-opacity to hide labels without affecting layout)
    styles.push({
      selector: 'edge',
      style: {
        'width': 'data(width)',
        'line-color': '#334155',
        'target-arrow-color': '#334155',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'label': 'data(label)',
        'text-opacity': showEdgeLabels ? 1 : 0,
        'font-size': '11px',
        'text-rotation': 'autorotate',
        'text-margin-y': -10,
        'color': '#1e293b',
        'text-background-color': 'transparent',
        'text-background-opacity': 0
      }
    });

    // Self-loop style
    styles.push({
      selector: 'edge[isSelfLoop]',
      style: {
        'curve-style': 'loop',
        'loop-direction': '0deg',
        'loop-sweep': '90deg',
        'control-point-step-size': 40
      }
    });

    // Highlight / dim styles
    styles.push({
      selector: '.highlighted',
      style: {
        'line-color': '#0f172a',
        'target-arrow-color': '#0f172a',
        'opacity': 1
      }
    });

    styles.push({
      selector: '.highlighted-node',
      style: {
        'opacity': 1
      }
    });

    styles.push({
      selector: '.dimmed',
      style: {
        'opacity': 0.2
      }
    });

    // Create Cytoscape instance
    this.cy = cytoscape({
      container: this.container,
      elements,
      style: styles,
      layout: this.getLayoutConfig(layoutMode, graphData, usePresetPositions),
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

  private formatProbability(p: number): string {
    // Convert to percentage and format with comma as decimal separator
    const percentage = (p * 100).toFixed(2);
    return percentage.replace('.', ',') + '%';
  }

  private getEdgeWidth(probability: number, isSelfLoop: boolean, allEdges: typeof import('./types').GraphData.edges): number {
    // Use separate scales for self-loops vs state transitions
    // This makes differences visible in both categories

    if (isSelfLoop) {
      // Self-loops typically have high probabilities (0.7-1.0)
      // Find min/max among self-loops for adaptive scaling
      const selfLoopProbs = allEdges.filter(e => e.source === e.target).map(e => e.p);
      const minSelfLoop = Math.min(...selfLoopProbs);
      const maxSelfLoop = Math.max(...selfLoopProbs);
      const range = maxSelfLoop - minSelfLoop || 0.1; // Avoid division by zero

      // Map to 2-6px range
      const normalized = (probability - minSelfLoop) / range;
      return Math.max(2, Math.min(6, 2 + 4 * normalized));
    } else {
      // Transitions typically have lower probabilities (0-0.3)
      // Find min/max among transitions for adaptive scaling
      const transitionProbs = allEdges.filter(e => e.source !== e.target).map(e => e.p);
      if (transitionProbs.length === 0) return 2;

      const minTransition = Math.min(...transitionProbs);
      const maxTransition = Math.max(...transitionProbs);
      const range = maxTransition - minTransition || 0.1;

      // Map to 1-5px range
      const normalized = (probability - minTransition) / range;
      return Math.max(1, Math.min(5, 1 + 4 * normalized));
    }
  }

  private getLayoutConfig(
    layoutMode: 'default' | 'connections' | 'deployer' | 'geo-level',
    graphData: GraphData,
    usePresetPositions: boolean
  ) {
    switch (layoutMode) {
      case 'default':
        // Use preset positions for n=5/15 or when preset positions were computed,
        // otherwise force-directed
        if (usePresetPositions || graphData.nodes.length === 5 || graphData.nodes.length === 15) {
          return { name: 'preset' };
        } else {
          return {
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
          };
        }

      case 'connections':
        // Force-directed layout, but use saved or preset positions if available
        if (usePresetPositions) {
          return { name: 'preset' };
        } else {
          return {
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
          };
        }

      case 'deployer': {
        // Cluster by deploying coalition
        const nodesByDeployer = new Map<string, string[]>();
        graphData.nodes.forEach(node => {
          const deployer = node.meta?.deploying_coalition || 'None';
          if (!nodesByDeployer.has(deployer)) {
            nodesByDeployer.set(deployer, []);
          }
          nodesByDeployer.get(deployer)!.push(node.id);
        });

        const groups = Array.from(nodesByDeployer.values()).map(ids => ({ leaves: ids }));

        return {
          name: 'preset',  // Use preset with our computed cluster positions
          fit: true,
          padding: 50
        };
      }

      case 'geo-level': {
        // Cluster by geoengineering level
        return {
          name: 'preset',  // Use preset with our computed cluster positions
          fit: true,
          padding: 50
        };
      }

      default:
        return { name: 'cose-bilkent', animate: false };
    }
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

  // Highlight a set of nodes and their connected edges; dim everything else
  highlightNodes(nodeIds: string[]) {
    if (!this.cy) return;
    // Clear previous
    this.cy.elements().removeClass('highlighted dimmed highlighted-node');

    const nodes = this.cy.nodes().filter((n) => nodeIds.includes(n.id()));
    if (nodes.length === 0) return;

    nodes.addClass('highlighted-node');

    const connectedEdges = nodes.connectedEdges();
    connectedEdges.addClass('highlighted');

    // Dim everything else
    this.cy.elements().not(nodes).not(connectedEdges).addClass('dimmed');
  }

  clearHighlights() {
    if (!this.cy) return;
    this.cy.elements().removeClass('highlighted dimmed highlighted-node');
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
      data: {
        geo_level: node.data('geo_level'),
        deploying_coalition: node.data('deploying_coalition')
      },
      meta: {
        geo_level: node.data('geo_level'),
        deploying_coalition: node.data('deploying_coalition')
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
