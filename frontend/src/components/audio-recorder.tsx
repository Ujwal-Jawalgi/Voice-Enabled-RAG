"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Square, Loader2, Volume2, AlertCircle } from "lucide-react";

interface AudioRecorderProps {
  onRecordingComplete: (payload: { audio_base64?: string; text?: string }) => void;
  isLoading: boolean;
}

export function AudioRecorder({ onRecordingComplete, isLoading }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Stop recording and cleanup
  const stopAll = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {}
      recognitionRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      try {
        mediaRecorderRef.current.stop();
      } catch {}
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsRecording(false);
    setAudioLevel(0);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAll();
    };
  }, [stopAll]);

  const startRecording = async () => {
    setMicError(null);
    setLiveTranscript("");
    chunksRef.current = [];

    // 1. Request microphone stream
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;
    } catch (err: any) {
      console.error("Microphone access error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setMicError("Microphone permission was denied. Please allow microphone access in your browser address bar.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setMicError("No microphone found on your device. Please connect a microphone.");
      } else {
        setMicError("Could not access microphone: " + (err.message || "Unknown error"));
      }
      return;
    }

    // 2. Setup Audio Visualizer Analyzer
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        const audioCtx = new AudioCtx();
        audioContextRef.current = audioCtx;
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        analyserRef.current = analyser;
        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const checkLevel = () => {
          if (analyserRef.current) {
            analyserRef.current.getByteFrequencyData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
              sum += dataArray[i];
            }
            const avg = sum / dataArray.length;
            setAudioLevel(Math.min(100, Math.round((avg / 128) * 100)));
          }
          animFrameRef.current = requestAnimationFrame(checkLevel);
        };
        checkLevel();
      }
    } catch (e) {
      console.warn("AudioContext visualizer setup skipped", e);
    }

    // 3. Optional: Client-side Web Speech Recognition for instant zero-latency feedback
    let speechRecognizedText = "";
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      try {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;

        recognition.onresult = (event: any) => {
          let currentText = "";
          for (let i = 0; i < event.results.length; i++) {
            currentText += event.results[i][0].transcript;
          }
          if (currentText) {
            speechRecognizedText = currentText;
            setLiveTranscript(currentText);
          }
        };

        recognition.onerror = (event: any) => {
          console.warn("Web Speech recognition warning:", event.error);
        };

        recognition.start();
        recognitionRef.current = recognition;
      } catch (e) {
        console.warn("SpeechRecognition init skipped", e);
      }
    }

    // 4. Setup MediaRecorder for backend audio payload
    try {
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "";

      const options = mimeType ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const actualMime = mimeType || "audio/webm";
        const audioBlob = new Blob(chunksRef.current, { type: actualMime });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const base64Audio = reader.result as string;
          const finalPayload: { audio_base64?: string; text?: string } = {
            audio_base64: base64Audio,
          };
          if (speechRecognizedText.trim()) {
            finalPayload.text = speechRecognizedText.trim();
          }
          onRecordingComplete(finalPayload);
        };

        // Stop all media tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(250); // Slice every 250ms
      setIsRecording(true);
    } catch (err: any) {
      console.error("MediaRecorder start error:", err);
      setMicError("Failed to initialize recording recorder: " + err.message);
      stopAll();
    }
  };

  const stopRecording = () => {
    stopAll();
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="flex flex-col items-center justify-center gap-3 w-full max-w-md mx-auto">
      {/* Microphone Record Button */}
      <div className="relative flex items-center justify-center">
        {/* Pulsing ring indicator based on live voice audio level */}
        {isRecording && (
          <div
            className="absolute rounded-full bg-purple-500/30 transition-all duration-75 pointer-events-none"
            style={{
              width: `${80 + audioLevel * 0.8}px`,
              height: `${80 + audioLevel * 0.8}px`,
            }}
          />
        )}

        <button
          type="button"
          onClick={toggleRecording}
          disabled={isLoading && !isRecording}
          className={`relative z-10 flex h-20 w-20 items-center justify-center rounded-full transition-all duration-300 ${
            isRecording
              ? "bg-rose-600 hover:bg-rose-700 shadow-xl shadow-rose-500/40 scale-105"
              : isLoading
              ? "bg-zinc-200 dark:bg-zinc-800 cursor-not-allowed opacity-70"
              : "bg-gradient-to-tr from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 hover:scale-105 shadow-xl shadow-indigo-500/30"
          }`}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
        >
          {isLoading && !isRecording ? (
            <Loader2 className="h-8 w-8 text-zinc-400 animate-spin" />
          ) : isRecording ? (
            <Square className="h-7 w-7 text-white fill-current" />
          ) : (
            <Mic className="h-8 w-8 text-white" />
          )}
        </button>
      </div>

      {/* Voice Status & Live Transcription */}
      <div className="min-h-12 flex flex-col items-center justify-center text-center px-4">
        {isRecording ? (
          <div className="space-y-1 animate-in fade-in duration-200">
            <div className="flex items-center gap-2 text-rose-500 dark:text-rose-400 font-semibold text-sm">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping" />
              <Volume2 className="w-4 h-4" />
              Listening... Speak now (Tap to finish)
            </div>
            {liveTranscript ? (
              <p className="text-sm font-medium text-foreground bg-foreground/5 px-3 py-1.5 rounded-full border border-border/50 max-w-sm truncate">
                "{liveTranscript}"
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">Detecting audio...</p>
            )}
          </div>
        ) : isLoading ? (
          <p className="text-sm text-indigo-500 font-medium animate-pulse">
            Processing your question...
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">
            Tap mic and speak, or type your question below
          </p>
        )}
      </div>

      {/* Permission / Device Error Banner */}
      {micError && (
        <div className="w-full flex items-start gap-2.5 p-3 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400 text-xs animate-in fade-in">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium">{micError}</p>
          </div>
          <button
            type="button"
            onClick={() => setMicError(null)}
            className="text-muted-foreground hover:text-foreground shrink-0 font-bold"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
