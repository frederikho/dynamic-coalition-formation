export interface GraphNode {
  id: string;
  label: string;
  x?: number;
  y?: number;
  meta?: {
    index?: number;
    geo_level?: number;
    deploying_coalition?: string;
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
    num_players: number;
    num_states: number;
    num_transitions: number;
    expected_geo_level?: number | null;
    stationary_distribution?: Record<string, number> | null;
    mixing_time?: number | null;
    scenario_name?: string;
    scenario_description?: string;
    config: {
      power_rule: string;
      unanimity_required: boolean;
      min_power: number | null;
    };
    note?: string;
    file_metadata?: {
      [key: string]: any;
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
