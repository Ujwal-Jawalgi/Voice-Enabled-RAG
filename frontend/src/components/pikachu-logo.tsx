export function PikachuLogo({ className = "w-6 h-6" }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" className={className}>
      {/* Left Ear */}
      <path d="M 30 35 L 5 5 L 45 25 Z" fill="#F6D025" />
      {/* Left Ear Tip */}
      <path d="M 17 20 L 5 5 L 31 15 Z" fill="#222224" />
      
      {/* Right Ear */}
      <path d="M 70 35 L 95 5 L 55 25 Z" fill="#F6D025" />
      {/* Right Ear Tip */}
      <path d="M 83 20 L 95 5 L 69 15 Z" fill="#222224" />

      {/* Face */}
      <ellipse cx="50" cy="60" rx="40" ry="35" fill="#F6D025" />

      {/* Eyes */}
      <circle cx="33" cy="52" r="6" fill="#222224" />
      <circle cx="67" cy="52" r="6" fill="#222224" />
      
      {/* Eye Sparkles */}
      <circle cx="31" cy="50" r="2" fill="#FFFFFF" />
      <circle cx="65" cy="50" r="2" fill="#FFFFFF" />

      {/* Cheeks */}
      <circle cx="18" cy="68" r="7" fill="#E3350D" />
      <circle cx="82" cy="68" r="7" fill="#E3350D" />

      {/* Nose */}
      <circle cx="50" cy="58" r="1.5" fill="#222224" />

      {/* Mouth */}
      <path d="M 42 66 Q 46 70 50 65 Q 54 70 58 66" fill="none" stroke="#222224" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
