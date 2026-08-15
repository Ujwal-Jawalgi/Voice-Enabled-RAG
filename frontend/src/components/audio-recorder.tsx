"use client";

import { useState, useRef, useEffect } from "react";
import { Mic, Square, Loader2 } from "lucide-react";

interface AudioRecorderProps {
  onRecordingComplete: (base64Audio: string) => void;
  isLoading: boolean;
}

export function AudioRecorder({ onRecordingComplete, isLoading }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/wav" });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const base64Audio = reader.result as string;
          onRecordingComplete(base64Audio);
        };
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please check your permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="flex flex-col items-center justify-center gap-2">
      <button
        onClick={toggleRecording}
        disabled={isLoading && !isRecording}
        className={`relative flex h-20 w-20 items-center justify-center rounded-full transition-all duration-300 ${
          isRecording 
            ? "bg-purple-600 animate-pulse-ring hover:bg-purple-700" 
            : isLoading 
              ? "bg-zinc-200 dark:bg-zinc-800 cursor-not-allowed" 
              : "bg-indigo-600 hover:bg-indigo-700 hover:scale-105 shadow-lg shadow-indigo-500/30"
        }`}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
      >
        {isLoading && !isRecording ? (
          <Loader2 className="h-8 w-8 text-zinc-400 animate-spin" />
        ) : isRecording ? (
          <Square className="h-8 w-8 text-white fill-current" />
        ) : (
          <Mic className="h-8 w-8 text-white" />
        )}
      </button>
      <div className="h-6 mt-2 flex items-center justify-center">
        {isRecording && (
          <span className="text-sm font-medium text-purple-600 dark:text-purple-400 animate-pulse">
            Listening... (Tap to stop)
          </span>
        )}
      </div>
    </div>
  );
}
