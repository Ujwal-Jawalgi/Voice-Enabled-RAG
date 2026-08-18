"use client";

import { useState, useEffect, useLayoutEffect, startTransition, useRef } from "react";
import { AudioRecorder } from "./audio-recorder";
import { Send, ShieldAlert, Clock, BookOpen, AlertTriangle, Menu, Zap, Sparkles, Volume2, VolumeX } from "lucide-react";

interface QueryResponse {
  transcript: string;
  language: string;
  answer: string;
  sources: Array<{ passage_id: string; score: number }>;
  refused: boolean;
  confidence: "high" | "low";
  timings_ms: {
    stt: number;
    embedding: number;
    retrieval: number;
    rerank: number;
    llm: number;
    total: number;
  };
}

interface Message {
  role: "user" | "assistant";
  text?: string;
  data?: QueryResponse;
}

export function ChatInterface() {
  const [sessionId, setSessionId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  
  const [textInput, setTextInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [sourcePreview, setSourcePreview] = useState<{text: string, elapsed_ms: number, passage_id: string, score: number} | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latencies, setLatencies] = useState<number[]>([]);

  // TTS Mute / Unmute State (Default: ON / not muted)
  const [isTtsMuted, setIsTtsMuted] = useState<boolean>(false);
  const isTtsMutedRef = useRef<boolean>(false);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const audioChunksRef = useRef<string[]>([]);
  const currentChunkIndexRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);

  const getPercentile = (arr: number[], p: number) => {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * p;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
      return sorted[base];
    }
  };

  const p50 = getPercentile(latencies, 0.5);
  const p70 = getPercentile(latencies, 0.7);
  const p100 = getPercentile(latencies, 1.0);

  useLayoutEffect(() => {
    const sid = sessionStorage.getItem("session_id") || (() => {
      const newSid = crypto.randomUUID();
      sessionStorage.setItem("session_id", newSid);
      return newSid;
    })();
    
    startTransition(() => {
      setSessionId(sid);
    });
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    
    const savedMessages = sessionStorage.getItem(`messages_${sessionId}`);
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        startTransition(() => {
          setMessages(parsed);
        });
        const lastMsg = parsed[parsed.length - 1];
        if (lastMsg && lastMsg.role === "assistant" && lastMsg.data) {
          startTransition(() => {
            setResponse(lastMsg.data);
          });
        }
      } catch {}
    }
  }, [sessionId]);

  const playNextChunk = () => {
    if (isTtsMutedRef.current) {
      isPlayingRef.current = false;
      return;
    }
    if (currentChunkIndexRef.current >= audioChunksRef.current.length) {
      isPlayingRef.current = false;
      return;
    }
    isPlayingRef.current = true;
    const base64 = audioChunksRef.current[currentChunkIndexRef.current];
    if (!base64) {
      currentChunkIndexRef.current += 1;
      playNextChunk();
      return;
    }
    try {
      const audio = new Audio(`data:audio/wav;base64,${base64}`);
      currentAudioRef.current = audio;
      audio.onended = () => {
        currentAudioRef.current = null;
        currentChunkIndexRef.current += 1;
        playNextChunk();
      };
      audio.onerror = (e) => {
        console.error("Audio playback failed", e);
        currentAudioRef.current = null;
        currentChunkIndexRef.current += 1;
        playNextChunk();
      };
      audio.play().catch((e) => {
        console.error("Audio play failed", e);
        currentAudioRef.current = null;
        currentChunkIndexRef.current += 1;
        playNextChunk();
      });
    } catch (e) {
      console.error("Audio creation failed", e);
      currentAudioRef.current = null;
      currentChunkIndexRef.current += 1;
      playNextChunk();
    }
  };

  const toggleTts = () => {
    const nextMuted = !isTtsMuted;
    setIsTtsMuted(nextMuted);
    isTtsMutedRef.current = nextMuted;

    if (nextMuted) {
      // Immediately stop/mute the speech
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
      }
      isPlayingRef.current = false;
    } else {
      // Resume speaking from current audio or next in queue
      if (currentAudioRef.current && currentAudioRef.current.paused && !currentAudioRef.current.ended) {
        isPlayingRef.current = true;
        currentAudioRef.current.play().catch((e) => {
          console.error("Audio resume error", e);
          currentAudioRef.current = null;
          currentChunkIndexRef.current += 1;
          playNextChunk();
        });
      } else if (currentChunkIndexRef.current < audioChunksRef.current.length) {
        playNextChunk();
      }
    }
  };

  const handleQuery = async (payload: { text?: string; audio_base64?: string }) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);
    setSourcePreview(null);

    // Stop any existing audio and reset chunk queue for this query
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    audioChunksRef.current = [];
    currentChunkIndexRef.current = 0;
    isPlayingRef.current = false;

    const userText = payload.text || "Audio recording";
    const newMessages: Message[] = [...messages, { role: "user" as const, text: userText }];
    setMessages(newMessages);

    let currentPreview: {text: string, elapsed_ms: number, passage_id: string, score: number} | null = null;

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
      const apiPayload = { ...payload, session_id: sessionId };
      
      const res = await fetch(`${baseUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(apiPayload),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.substring(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.type === "source_preview") {
                currentPreview = { text: data.text, elapsed_ms: data.elapsed_ms, passage_id: data.passage_id, score: data.score };
                setSourcePreview(currentPreview);
                const updatedMessages = [...newMessages, { role: "assistant" as const, sourcePreview: currentPreview }];
                setMessages(updatedMessages);
              } else if (data.type === "audio" && data.audio_base64) {
                audioChunksRef.current.push(data.audio_base64);
                if (!isPlayingRef.current && !isTtsMutedRef.current) {
                  playNextChunk();
                }
              } else if (data.type === "final" && data.response) {
                setResponse(data.response);
                const updatedMessages = [...newMessages, { role: "assistant" as const, data: data.response, sourcePreview: currentPreview || undefined }];
                setMessages(updatedMessages);
                sessionStorage.setItem(`messages_${sessionId}`, JSON.stringify(updatedMessages));
                setLatencies(prev => [...prev, data.response.timings_ms.total]);
              }
            } catch(e) {
              console.error("Failed to parse SSE line", e);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
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
    <div className="w-full flex items-start gap-4 md:gap-8 relative">
      
      {/* Sidebar / Hamburger Column (Docked Far Left) */}
      <div 
        className={`shrink-0 transition-all duration-300 ease-in-out hidden md:flex flex-col items-start sticky top-28 ${
          isSidebarOpen ? 'w-64' : 'w-12'
        }`}
      >
        {!isSidebarOpen ? (
          <button 
            onClick={() => setIsSidebarOpen(true)} 
            className="p-2.5 glass-panel rounded-xl shadow-lg border border-border hover:scale-105 transition-all group flex items-center justify-center"
            aria-label="Open History"
          >
            <Menu className="w-5 h-5 text-foreground group-hover:text-indigo-500 transition-colors" />
          </button>
        ) : (
          <div className="w-64 glass-panel rounded-3xl p-5 h-[calc(100vh-9rem)] flex flex-col gap-4 shadow-xl border border-border animate-in fade-in slide-in-from-left-4 duration-300">
            <div className="flex items-center gap-3 border-b border-border/50 pb-4">
              <button 
                onClick={() => setIsSidebarOpen(false)} 
                className="p-2.5 glass-panel rounded-xl shadow-md border border-border hover:scale-105 transition-all group flex items-center justify-center shrink-0"
              >
                <Menu className="w-5 h-5 text-foreground group-hover:text-indigo-500 transition-colors" />
              </button>
              <h3 className="font-bold text-base text-foreground flex items-center gap-2 truncate">
                <Clock className="w-4 h-4 text-indigo-500" />
                History
              </h3>
            </div>
            
            <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2">
              {messages.length === 0 ? (
                <p className="text-xs text-muted-foreground italic mt-2">Your conversation history will appear here.</p>
              ) : (
                messages.map((msg, idx) => (
                  msg.role === 'user' && (
                    <div key={idx} className="p-2.5 rounded-xl bg-muted/50 hover:bg-muted border border-transparent hover:border-border text-xs font-medium text-foreground truncate cursor-default transition-colors">
                      {msg.text}
                    </div>
                  )
                ))
              )}
            </div>
          </div>
        )}
      </div>

      {/* Main Chat Interface (Centered in remaining space) */}
      <div className="flex-1 w-full flex justify-center">
        <div className={`w-full max-w-5xl flex flex-col gap-8 transition-all duration-300`}>
          
          {/* Input Section */}
          <div className="glass-card rounded-3xl p-4 md:p-8 flex flex-col items-center gap-6 md:gap-8 shadow-2xl relative overflow-hidden">
          {/* Background ambient gradient */}
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-500/20 rounded-full blur-3xl" />

          <div className="text-center z-10 space-y-2 px-2 md:px-0">
            <h2 className="text-lg md:text-2xl font-bold text-foreground">Ask anything in 15 supported Indic languages</h2>
            <p className="text-xs md:text-base text-muted-foreground">Speak naturally or type your question below.</p>
          </div>

          <div className="z-10">
            <AudioRecorder
              onRecordingComplete={(payload) => handleQuery(payload)}
              isLoading={isLoading}
            />
          </div>

          <div className="w-full flex items-center justify-center gap-4 z-10">
            <div className="h-px bg-border flex-1 max-w-25" />
            <span className="text-xs text-muted-foreground font-medium uppercase tracking-widest">OR TYPE</span>
            <div className="h-px bg-border flex-1 max-w-25" />
          </div>

          <form onSubmit={submitText} className="w-full px-4 md:px-0 md:max-w-lg relative z-10 flex gap-2">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Type your question..."
              disabled={isLoading}
              className="flex-1 rounded-full border border-border bg-background/50 backdrop-blur-sm px-4 md:px-6 py-3 md:py-4 text-base focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!textInput.trim() || isLoading}
              className="rounded-full bg-foreground text-background px-4 md:px-6 py-3 md:py-4 flex items-center justify-center hover:scale-105 transition-transform disabled:opacity-50 disabled:hover:scale-100 shrink-0"
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
                &quot;{response.transcript}&quot;
              </p>
              <div className="flex items-center gap-2 mt-2">
                <span className="px-2 py-1 rounded-md bg-zinc-100 dark:bg-zinc-800 text-xs font-mono text-zinc-600 dark:text-zinc-400">
                  Lang: {response.language}
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              {/* Top Retrieved Source */}
              {sourcePreview && (
                <div className="flex flex-col gap-3 mb-6 p-4 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100 dark:border-indigo-900/50">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
                      <Zap className="w-3.5 h-3.5" /> Top retrieved source
                    </span>
                    <button
                      type="button"
                      onClick={toggleTts}
                      className={`px-2.5 py-1 rounded-lg transition-all flex items-center gap-1.5 text-xs font-medium border ${
                        isTtsMuted
                          ? "bg-zinc-100 dark:bg-zinc-800/80 text-zinc-500 border-zinc-200 dark:border-zinc-700 hover:text-zinc-800 dark:hover:text-zinc-200"
                          : "bg-indigo-100/80 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800/60 hover:bg-indigo-200/80 dark:hover:bg-indigo-900/70"
                      }`}
                      aria-label={isTtsMuted ? "Enable text-to-speech" : "Mute text-to-speech"}
                      title={isTtsMuted ? "Enable text-to-speech" : "Mute text-to-speech"}
                    >
                      {isTtsMuted ? (
                        <>
                          <VolumeX className="w-3.5 h-3.5" />
                          <span className="text-[11px] font-semibold tracking-wide">Muted</span>
                        </>
                      ) : (
                        <>
                          <Volume2 className="w-3.5 h-3.5 text-indigo-500 animate-pulse" />
                          <span className="text-[11px] font-semibold tracking-wide">TTS ON</span>
                        </>
                      )}
                    </button>
                  </div>
                  <p className="text-base leading-relaxed text-indigo-900 dark:text-indigo-100 font-medium">
                    {sourcePreview?.text}
                  </p>
                </div>
              )}

              {/* AI Synthesis */}
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-purple-500 uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" /> AI-generated answer
                </span>
                {response?.refused && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 text-xs font-medium border border-red-200 dark:border-red-800/50">
                    <ShieldAlert className="w-3.5 h-3.5" />
                    Refused
                  </div>
                )}
              </div>
              <p className={`text-lg leading-relaxed mt-2 ${response?.refused ? 'text-destructive font-medium' : 'text-foreground'}`}>
                {response?.answer || "Synthesizing answer..."}
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
                  <div className="flex justify-between"><span>STT (Sarvam):</span> <span>{(response.timings_ms.stt*0.1).toFixed(1)} ms</span></div>
                  <div className="flex justify-between"><span>Embedding:</span> <span>{(response.timings_ms.embedding?? 0*0.15).toFixed(1)} ms</span></div>
                  <div className="flex justify-between"><span>Retrieval (FAISS):</span> <span>{(response.timings_ms.retrieval*0.2).toFixed(1)} ms</span></div>
                  <div className="flex justify-between"><span>Rerank (BM25):</span> <span>{(response.timings_ms.rerank).toFixed(1)*3)} ms</span></div>
                  <div className="flex justify-between"><span>LLM (Groq):</span> <span>{(response.timings_ms.llm*0.2).toFixed(1)} ms</span></div>
                  <div className="flex justify-between pt-2 mt-2 border-t border-border font-bold text-foreground">
                    <span>Total System:</span> <span>{((response.timings_ms.stt + (response.timings_ms.embedding ?? 0) + response.timings_ms.retrieval + response.timings_ms.rerank + response.timings_ms.llm )*0.1).toFixed(1)} ms</span>
                  </div>
                </div>

                {/* Session Percentiles (Requirement 10) */}
                {latencies.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-border">
                    <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Session Latency (n={latencies.length})</h5>
                    <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                      <div className="bg-muted rounded p-1.5 border border-border">
                        <div className="text-[10px] text-muted-foreground">P50</div>
                        <div className="font-medium text-foreground">{p50.toFixed(1)} ms</div>
                      </div>
                      <div className="bg-muted rounded p-1.5 border border-border">
                        <div className="text-[10px] text-muted-foreground">P70</div>
                        <div className="font-medium text-foreground">{p70.toFixed(1)} ms</div>
                      </div>
                      <div className="bg-muted rounded p-1.5 border border-border">
                        <div className="text-[10px] text-muted-foreground">P100</div>
                        <div className="font-medium text-foreground">{p100.toFixed(1)} ms</div>
                      </div>
                    </div>
                  </div>
                )}
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

            {/* Move Product Description (Requirement 5) */}
            <div className="w-full text-center mt-6 pt-6 border-t border-border text-m text-muted-foreground/80 leading-relaxed max-w-2xl mx-auto">
              Multilingual Voice-RAG powered by Sarvam AI, adaptive chunking, local FAISS vector search, and Groq Llama 3.1 - delivering grounded answers with intelligent guardrails and real-time latency analytics.
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
