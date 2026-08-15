"use client";

import { useState } from "react";
import { AudioRecorder } from "./audio-recorder";
import { Send, ShieldCheck, ShieldAlert, Clock, BookOpen, AlertTriangle } from "lucide-react";

interface QueryResponse {
  transcript: string;
  language: string;
  answer: string;
  sources: Array<{ passage_id: string; score: number }>;
  refused: boolean;
  confidence: "high" | "low";
  timings_ms: {
    stt: number;
    retrieval: number;
    rerank: number;
    llm: number;
    total: number;
  };
}

export function ChatInterface() {
  const [textInput, setTextInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async (payload: { text?: string; audio_base64?: string }) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${baseUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const submitText = (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;
    handleQuery({ text: textInput });
    setTextInput("");
  };

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col gap-8">
      
      {/* Input Section */}
      <div className="glass-card rounded-3xl p-8 flex flex-col items-center gap-8 shadow-2xl relative overflow-hidden">
        {/* Background ambient gradient */}
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-500/20 rounded-full blur-3xl" />

        <div className="text-center z-10 space-y-2">
          <h2 className="text-2xl font-bold text-foreground">Ask anything in English, Hindi, or Kannada</h2>
          <p className="text-muted-foreground">Speak naturally or type your question below.</p>
        </div>

        <div className="z-10">
          <AudioRecorder 
            onRecordingComplete={(base64) => handleQuery({ audio_base64: base64 })} 
            isLoading={isLoading} 
          />
        </div>

        <div className="w-full flex items-center justify-center gap-4 z-10">
          <div className="h-px bg-border flex-1 max-w-[100px]" />
          <span className="text-xs text-muted-foreground font-medium uppercase tracking-widest">OR TYPE</span>
          <div className="h-px bg-border flex-1 max-w-[100px]" />
        </div>

        <form onSubmit={submitText} className="w-full max-w-lg relative z-10 flex gap-2">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Type your question..."
            disabled={isLoading}
            className="flex-1 rounded-full border border-border bg-background/50 backdrop-blur-sm px-6 py-4 text-base focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!textInput.trim() || isLoading}
            className="rounded-full bg-foreground text-background px-6 flex items-center justify-center hover:scale-105 transition-transform disabled:opacity-50 disabled:hover:scale-100"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="glass-panel rounded-2xl p-6 border-destructive/50 flex items-start gap-4 text-destructive">
          <AlertTriangle className="w-6 h-6 shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold">Error communicating with server</h3>
            <p className="text-sm opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* Results Section */}
      {response && (
        <div className="glass-panel rounded-3xl p-8 flex flex-col gap-6 shadow-2xl animate-in fade-in slide-in-from-bottom-8 duration-500">
          
          <div className="flex flex-col gap-2 border-b border-border pb-6">
            <span className="text-xs font-semibold text-indigo-500 uppercase tracking-wider">Detected Transcript</span>
            <p className="text-xl font-medium text-foreground">
              "{response.transcript}"
            </p>
            <div className="flex items-center gap-2 mt-2">
              <span className="px-2 py-1 rounded-md bg-zinc-100 dark:bg-zinc-800 text-xs font-mono text-zinc-600 dark:text-zinc-400">
                Lang: {response.language}
              </span>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-purple-500 uppercase tracking-wider">Answer</span>
              {response.refused ? (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 text-xs font-medium border border-red-200 dark:border-red-800/50">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Refused
                </div>
              ) : response.confidence === "high" ? (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 text-xs font-medium border border-emerald-200 dark:border-emerald-800/50">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  High Confidence
                </div>
              ) : (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 text-xs font-medium border border-amber-200 dark:border-amber-800/50">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Low Confidence
                </div>
              )}
            </div>
            
            <p className={`text-lg leading-relaxed ${response.refused ? 'text-destructive font-medium' : 'text-foreground'}`}>
              {response.answer}
            </p>
          </div>

          {/* Metrics & Sources Accordion */}
          <div className="mt-4 pt-6 border-t border-border grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Clock className="w-4 h-4" />
                <h4 className="text-sm font-semibold uppercase tracking-wider">Latency Breakdown</h4>
              </div>
              <div className="space-y-2 text-sm font-mono text-zinc-600 dark:text-zinc-400">
                <div className="flex justify-between"><span>STT (Sarvam):</span> <span>{response.timings_ms.stt.toFixed(1)} ms</span></div>
                <div className="flex justify-between"><span>Retrieval (FAISS):</span> <span>{response.timings_ms.retrieval.toFixed(1)} ms</span></div>
                <div className="flex justify-between"><span>Rerank (BM25):</span> <span>{response.timings_ms.rerank.toFixed(1)} ms</span></div>
                <div className="flex justify-between"><span>LLM (Groq):</span> <span>{response.timings_ms.llm.toFixed(1)} ms</span></div>
                <div className="flex justify-between pt-2 mt-2 border-t border-border font-bold text-foreground">
                  <span>Total System:</span> <span>{response.timings_ms.total.toFixed(1)} ms</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-2 text-muted-foreground">
                <BookOpen className="w-4 h-4" />
                <h4 className="text-sm font-semibold uppercase tracking-wider">Sources Retrieved</h4>
              </div>
              {response.sources.length === 0 ? (
                <p className="text-sm text-muted-foreground italic">No sources retrieved.</p>
              ) : (
                <ul className="space-y-2">
                  {response.sources.map((s, idx) => (
                    <li key={idx} className="flex justify-between text-sm bg-background/50 px-3 py-2 rounded-lg border border-border">
                      <span className="font-mono text-indigo-500 truncate mr-4" title={s.passage_id}>{s.passage_id}</span>
                      <span className="text-muted-foreground tabular-nums">{s.score.toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
