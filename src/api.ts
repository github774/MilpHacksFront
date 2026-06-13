export interface ApiSimulationResult {
  riskScore: number;
  empathyScore: number;
  attentionScore: number;
  sentimentScore: number;
  harmScore: number;
  supportScore: number;
  saferRewrite: string;
  affectedGroups: {
    name: string;
    impact: string;
    severity: 'high' | 'medium' | 'low';
  }[];
  predictedEmotions: {
    name: string;
    percentage: number;
    color: string;
  }[];
  extractedTranscript?: string | null;
  mediaAnalysis?: {
    method: string;
    tone: string;
    topic: string;
    duration: number;
    segments: { start: number; end: number; text: string }[];
    caption: string;
  } | null;
  rawSimulation: {
    agents: {
      exposure_id: number;
      persona_index: number;
      wave: number;
      exposed_by_index: number | null;
      sampled_action: string;
      uuid: string;
      archetype: string;
    }[];
    share_edges: {
      from_exposure_id: number;
      from_index: number;
      from_uuid: string;
      from_archetype: string;
      from_occupation: string;
      to_index: number;
      to_uuid: string;
      to_archetype: string;
      to_occupation: string;
      share_type: string;
      wave: number;
    }[];
  };
}

export interface MediaFile {
  name: string;
  type: string;
  base64: string;
}

export async function simulateContent(
  text: string,
  selectedGroups: string[],
  mediaFile?: MediaFile | null
): Promise<ApiSimulationResult> {
  const response = await fetch("http://localhost:8000/api/simulate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      selected_groups: selectedGroups,
      n_seeds: 50,
      max_waves: 4,
      sharpness: 2.5,
      media_name: mediaFile?.name || null,
      media_type: mediaFile?.type || null,
      media_data: mediaFile?.base64 || null,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || "Failed to run simulation backend.");
  }

  return response.json();
}
