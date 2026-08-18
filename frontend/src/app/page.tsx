import { ChatInterface } from "@/components/chat-interface";
import { ThemeToggle } from "@/components/theme-toggle";
import { Zap } from "lucide-react";
import { PikachuLogo } from "@/components/pikachu-logo";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black font-sans selection:bg-indigo-500/30">
      
      {/* Premium Header */}
      <header className="w-full border-b border-border bg-white/50 dark:bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="w-full px-6 md:px-12 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center">
              <PikachuLogo className="w-12 h-12 drop-shadow-md" />
            </div>
            <span className="font-extrabold text-2xl tracking-tight">Pikachu</span>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-400 text-xs font-semibold uppercase tracking-widest">
              <Zap className="w-3.5 h-3.5" />
              HH Goa 2026
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full px-6 md:px-12 py-12 md:py-20 flex flex-col items-center">
        
        <div className="text-center space-y-4 mb-16 relative">
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">
            Speak naturally. <br className="hidden md:block" />
            <span className="text-gradient">Get grounded answers.</span>
          </h1>
        </div>

        <ChatInterface />

      </main>

      {/* Premium Footer */}
      <footer className="w-full border-t border-border mt-auto">
        <div className="w-full px-6 md:px-12 py-8 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>© 2026 Team Pikachu, HH Goa (Task 2)</p>
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
