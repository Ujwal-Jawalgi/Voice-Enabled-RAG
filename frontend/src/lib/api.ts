export interface Timings {
  stt: number;
  retrieval: number;
  rerank: number;
  llm: number;
  total: number;
}

export interface Source {
  passage_id: string;
  score: number;
}

export interface QueryRequest {
  audio_base64?: string;
  text?: string;
}

export interface QueryResponse {
  transcript: string;
  language: string;
  answer: string;
  sources: Source[];
  refused: boolean;
  confidence: "high" | "low";
  timings_ms: Timings;
}

export async function queryRAG(req: QueryRequest): Promise<QueryResponse> {
  // Simulate network latency
  await new Promise(resolve => setTimeout(resolve, 800));

  // Mock response based on the contract
  return {
    transcript: req.text || "Mocked speech transcript from audio",
    language: "hi",
    answer: "This is a mocked RAG generated answer grounded in the retrieved context.",
    sources: [
      { passage_id: "MSMARCO-XI-HI-12345", score: 0.89 },
      { passage_id: "MSMARCO-XI-HI-67890", score: 0.75 }
    ],
    refused: false,
    confidence: "high",
    timings_ms: {
      stt: 120,
      retrieval: 15,
      rerank: 25,
      llm: 35,
      total: 195
    }
  };
}
