export interface GraphNode {
  id: string;
  label: string;
  x?: number;
  y?: number;
  meta?: {
    index?: number;
    geo_level?: number;
    [key: string]: any;
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  p: number;
  meta?: {
    is_self_loop?: boolean;
    [key: string]: any;
  };
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: {
    profile_path: string;
    num_states: number;
    num_transitions: number;
    config: {
      power_rule: string;
      unanimity_required: boolean;
      min_power: number | null;
    };
  };
}

export interface Profile {
  name: string;
  path: string;
  filename: string;
}

export interface ProfilesResponse {
  profiles: Profile[];
  error?: string;
}
