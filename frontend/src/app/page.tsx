import { ChatInterface } from "@/components/chat-interface";
import { Mic, Zap } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black font-sans selection:bg-indigo-500/30">
      
      {/* Premium Header */}
      <header className="w-full border-b border-border bg-white/50 dark:bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Mic className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight">VoiceRAG</span>
          </div>
          
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-400 text-xs font-semibold uppercase tracking-widest">
            <Zap className="w-3.5 h-3.5" />
            HH Goa 2026
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-12 md:py-20 flex flex-col items-center">
        
        <div className="text-center space-y-4 mb-16 relative">
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">
            Speak naturally. <br className="hidden md:block" />
            <span className="text-gradient">Get grounded answers.</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Multilingual Voice-RAG powered by Sarvam AI, local FAISS vector search, and Groq's Llama 3.1.
          </p>
        </div>

        <ChatInterface />

      </main>

      {/* Premium Footer */}
      <footer className="w-full border-t border-border mt-auto">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>© 2026 HH Goa Hackathon Task 2</p>
          <div className="flex items-center gap-6">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
              Backend Online
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
              API Connected
            </span>
          </div>
        </div>
      </footer>
      
    </div>
  );
}
