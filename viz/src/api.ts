import type { GraphData, ProfilesResponse } from './types';

const API_BASE = 'http://127.0.0.1:8000';

export async function fetchProfiles(): Promise<ProfilesResponse> {
  const response = await fetch(`${API_BASE}/profiles`);
  if (!response.ok) {
    throw new Error(`Failed to fetch profiles: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchGraph(params: {
  profile: string;
}): Promise<GraphData> {
  const url = new URL(`${API_BASE}/graph`);
  url.searchParams.set('profile', params.profile);

  const response = await fetch(url.toString());
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Failed to fetch graph');
  }
  return response.json();
}
