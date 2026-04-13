import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Pause, FastForward, Rewind, 
  Volume2, RefreshCw, ChevronRight, 
  Languages, User, Search
} from 'lucide-react';
import './index.css';

const API_BASE = '/api';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [words, setWords] = useState([]);
  const [signData, setSignData] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const containerRef = useRef(null);
  const timerRef = useRef(null);

  // Quick phrases from the original project context
  const quickPhrases = [
    "HELLO", "THANK YOU", "GOOD MORNING", "PLEASE HELP ME", 
    "I WANT WATER", "SEE YOU LATER", "I LOVE YOU", "STOP"
  ];

  const handleConvert = async (textToConvert = inputText) => {
    if (!textToConvert.trim()) return;
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);
    setCurrentIndex(0);

    try {
      const response = await axios.post(`${API_BASE}/text-to-sign`, { 
        text: textToConvert 
      });
      if (response.data.success) {
        setWords(response.data.words);
        setSignData(response.data.sign_data);
      }
    } catch (err) {
      setError("Failed to fetch sign data. Is the backend running?");
    } finally {
      setIsLoading(false);
    }
  };

  const speak = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = playbackSpeed;
      window.speechSynthesis.speak(utterance);
    }
  };

  useEffect(() => {
    if (isPlaying && words.length > 0) {
      const interval = 1200 / playbackSpeed;
      
      timerRef.current = setInterval(() => {
        setCurrentIndex(prev => {
          if (prev >= words.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, interval);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, words, playbackSpeed]);

  const togglePlay = () => {
    if (currentIndex >= words.length - 1) {
      setCurrentIndex(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setInputText('');
    setWords([]);
    setSignData({});
    setCurrentIndex(0);
    setIsPlaying(false);
  };

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="logo-group">
          <div className="logo-icon">🤟</div>
          <h1>Sign<span>AI</span> <small>Motion</small></h1>
        </div>
        <div className="header-actions">
          <button className="icon-btn"><Languages size={18} /></button>
          <button className="icon-btn"><User size={18} /></button>
        </div>
      </header>

      <main className="main-content">
        <section className="input-section">
          <div className="glass-panel search-bar">
            <input 
              type="text" 
              placeholder="Type your message here... e.g. Hello how are you"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleConvert()}
            />
            <button className="primary-btn" onClick={() => handleConvert()}>
              <RefreshCw size={18} className={isLoading ? 'spin' : ''} />
              Convert
            </button>
          </div>
          
          <div className="quick-chips">
            {quickPhrases.map(phrase => (
              <button 
                key={phrase} 
                className="chip"
                onClick={() => {
                  setInputText(phrase);
                  handleConvert(phrase);
                }}
              >
                {phrase}
              </button>
            ))}
          </div>
        </section>

        <section className="visualization-section">
          <div className="stage-container">
            <AnimatePresence mode='wait'>
              {words.length > 0 ? (
                <motion.div 
                  key={words[currentIndex]}
                  initial={{ opacity: 0, scale: 0.9, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 1.1, y: -20 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="gesture-card highlight"
                >
                  <div className="media-wrapper">
                    {signData[words[currentIndex]]?.animation ? (
                      <video 
                        src={`/static/${signData[words[currentIndex]].animation}`} 
                        autoPlay loop muted playsInline 
                      />
                    ) : (
                      <div className="fallback-placeholder">
                        <div className="pulse-circle"></div>
                        <span>{words[currentIndex]?.charAt(0).toUpperCase()}</span>
                      </div>
                    )}
                  </div>
                  <div className="card-info">
                    <h2>{words[currentIndex]}</h2>
                    <p>{signData[words[currentIndex]]?.description || "Visualizing gesture..."}</p>
                  </div>
                </motion.div>
              ) : (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="empty-state"
                >
                  <div className="empty-icon">🎥</div>
                  <h3>Ready to Translate</h3>
                  <p>Enter text above to see the animated sign language sequence.</p>
                </motion.div>
              )}
            </AnimatePresence>

            {words.length > 0 && (
              <div className="controls-bar glass-panel">
                <div className="timeline-info">
                  <span className="current-word">{currentIndex + 1}</span>
                  <span className="separator">/</span>
                  <span className="total-words">{words.length}</span>
                </div>

                <div className="main-controls">
                  <button className="ctrl-btn" onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}>
                    <Rewind />
                  </button>
                  <button className="play-btn" onClick={togglePlay}>
                    {isPlaying ? <Pause size={32} fill="currentColor" /> : <Play size={32} fill="currentColor" />}
                  </button>
                  <button className="ctrl-btn" onClick={() => setCurrentIndex(Math.min(words.length - 1, currentIndex + 1))}>
                    <FastForward />
                  </button>
                </div>

                <div className="action-controls">
                  <button className="ctrl-btn" onClick={() => speak(words[currentIndex])}>
                    <Volume2 />
                  </button>
                  <select 
                    value={playbackSpeed} 
                    onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                    className="speed-selector"
                  >
                    <option value="0.5">0.5x</option>
                    <option value="1">1.0x</option>
                    <option value="1.5">1.5x</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          <div className="timeline-strip">
            {words.map((word, idx) => (
              <div 
                key={`${word}-${idx}`} 
                className={`timeline-item ${idx === currentIndex ? 'active' : ''} ${idx < currentIndex ? 'completed' : ''}`}
                onClick={() => setCurrentIndex(idx)}
              >
                <div className="item-label">{word}</div>
                <div className="item-dot"></div>
              </div>
            ))}
          </div>
        </section>
      </main>

      <style>{`
        .app-container {
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .navbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding-bottom: 1.5rem;
          border-bottom: 2px solid var(--border-soft);
        }

        .logo-group {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .logo-icon {
          font-size: 2.5rem;
          background: linear-gradient(135deg, var(--primary), #818cf8);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        h1 {
          font-size: 1.8rem;
          font-weight: 800;
          letter-spacing: -0.5px;
        }

        h1 span { color: var(--primary); }
        h1 small {
          font-size: 0.8rem;
          background: var(--primary);
          color: white;
          padding: 2px 8px;
          border-radius: 6px;
          margin-left: 8px;
          vertical-align: middle;
        }

        .icon-btn {
          background: rgba(255,255,255,0.05);
          border: 1px solid var(--border-soft);
          color: var(--text-muted);
          padding: 0.6rem;
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .icon-btn:hover {
          background: rgba(255,255,255,0.1);
          color: white;
        }

        .main-content {
          display: grid;
          grid-template-columns: 350px 1fr;
          gap: 2.5rem;
        }

        .glass-panel {
          background: var(--bg-card);
          backdrop-filter: var(--glass-effect);
          border: 1px solid var(--border-soft);
          border-radius: 24px;
          padding: 1.5rem;
        }

        .search-bar {
          display: flex;
          gap: 0.5rem;
          padding: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .search-bar input {
          flex: 1;
          background: transparent;
          border: none;
          color: white;
          padding: 0.75rem;
          font-size: 1rem;
        }

        .search-bar input:focus { outline: none; }

        .primary-btn {
          background: var(--primary);
          color: white;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 16px;
          font-weight: 700;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
          transition: transform 0.2s;
        }

        .primary-btn:hover { transform: scale(1.02); }

        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from {transform: rotate(0deg)} to {transform: rotate(360deg)} }

        .quick-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem;
        }

        .chip {
          background: rgba(255,255,255,0.05);
          border: 1px solid var(--border-soft);
          color: var(--text-muted);
          padding: 0.5rem 1rem;
          border-radius: 99px;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .chip:hover {
          background: rgba(59, 130, 246, 0.1);
          border-color: var(--primary);
          color: var(--primary);
        }

        .visualization-section {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .stage-container {
          position: relative;
          min-height: 500px;
          background: rgba(0,0,0,0.2);
          border-radius: 32px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 2rem;
          border: 1px solid var(--border-soft);
        }

        .gesture-card {
          width: 100%;
          max-width: 450px;
          background: rgba(255,255,255,0.03);
          border-radius: 28px;
          padding: 1.5rem;
          text-align: center;
          border: 1px solid rgba(255,255,255,0.1);
        }

        .media-wrapper {
          width: 100%;
          aspect-ratio: 1;
          background: #000;
          border-radius: 20px;
          overflow: hidden;
          margin-bottom: 1.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .media-wrapper video {
          width: 100%;
          height: 100%;
          object-fit: contain;
        }

        .fallback-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
        }

        .pulse-circle {
          width: 100px;
          height: 100px;
          border-radius: 50%;
          background: var(--primary);
          opacity: 0.2;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0% { transform: scale(1); opacity: 0.2; }
          50% { transform: scale(1.2); opacity: 0.4; }
          100% { transform: scale(1); opacity: 0.2; }
        }

        .card-info h2 {
          font-size: 2.2rem;
          text-transform: uppercase;
          letter-spacing: 2px;
          margin-bottom: 0.5rem;
          color: var(--primary);
        }

        .card-info p {
          color: var(--text-muted);
          font-style: italic;
        }

        .controls-bar {
          position: absolute;
          bottom: 2rem;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 2rem;
          padding: 1rem 2.5rem;
        }

        .main-controls {
          display: flex;
          align-items: center;
          gap: 1.5rem;
        }

        .play-btn {
          width: 70px;
          height: 70px;
          border-radius: 50%;
          background: var(--primary);
          border: none;
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
        }

        .ctrl-btn {
          background: none;
          border: none;
          color: var(--text-muted);
          cursor: pointer;
          transition: color 0.2s;
        }

        .ctrl-btn:hover { color: white; }

        .speed-selector {
          background: rgba(255,255,255,0.05);
          border: 1px solid var(--border-soft);
          color: white;
          padding: 4px 8px;
          border-radius: 8px;
        }

        .timeline-strip {
          display: flex;
          gap: 1rem;
          overflow-x: auto;
          padding: 1rem 0;
        }

        .timeline-item {
          min-width: 120px;
          padding: 1rem;
          background: rgba(255,255,255,0.03);
          border-radius: 16px;
          text-align: center;
          cursor: pointer;
          opacity: 0.5;
          transition: all 0.3s;
          border: 1px solid var(--border-soft);
        }

        .timeline-item.active {
          opacity: 1;
          background: rgba(59, 130, 246, 0.1);
          border-color: var(--primary);
          transform: scale(1.05);
        }

        .timeline-item.completed {
          opacity: 0.8;
          border-color: var(--accent);
        }

        .item-label {
          font-size: 0.75rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
        }

        .item-dot {
          width: 8px;
          height: 8px;
          background: #475569;
          border-radius: 50%;
          margin: 0 auto;
        }

        .active .item-dot { background: var(--primary); box-shadow: 0 0 10px var(--primary); }
        .completed .item-dot { background: var(--accent); }

        .empty-state {
          text-align: center;
          color: var(--text-muted);
        }

        .empty-icon { font-size: 5rem; margin-bottom: 1rem; opacity: 0.2; }

        @media (max-width: 1024px) {
          .main-content { grid-template-columns: 1fr; }
          .app-container { padding: 1rem; }
        }
      `}</style>
    </div>
  );
}
