import React, { useRef, useEffect, useState, useCallback } from "react";
import "./App.css";

const API = "http://localhost:8000";
const CAPTURE_INTERVAL = 300;

const SIGN_EMOJIS = {
  Hello: "👋", "Thank You": "🙏", Yes: "✅", No: "❌", Please: "🤲",
  Help: "🆘", Water: "💧", Food: "🍽️", Home: "🏠", Name: "🪪",
  Good: "👍", Bad: "👎", Stop: "✋", Come: "👉", Go: "🚶",
  I: "👆", You: "👇", Love: "❤️", India: "🇮🇳", Emergency: "🚨"
};

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [mode, setMode] = useState("sign2text");
  const [prediction, setPrediction] = useState({ sign: "", confidence: 0, hand_detected: false });
  const [sentence, setSentence] = useState([]);
  const [textInput, setTextInput] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [signs, setSigns] = useState([]);
  const [lang, setLang] = useState("en");
  const [isRunning, setIsRunning] = useState(false);
  const [lastAdded, setLastAdded] = useState("");
  const [holdCount, setHoldCount] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/signs`)
      .then(r => r.json())
      .then(d => setSigns(d.signs))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (mode === "sign2text") {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          if (videoRef.current) videoRef.current.srcObject = stream;
        })
        .catch(console.error);
    }
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      }
    };
  }, [mode]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
  }, []);

  const predict = useCallback(async () => {
    const frame = captureFrame();
    if (!frame) return;

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frame })
      });
      const data = await res.json();
      setPrediction(data);

      if (data.sign && data.confidence > 0.75) {
        if (data.sign === lastAdded) {
          setHoldCount(prev => {
            if (prev >= 3) {
              setSentence(s => [...s, data.sign]);
              setLastAdded("");
              return 0;
            }
            return prev + 1;
          });
        } else {
          setLastAdded(data.sign);
          setHoldCount(1);
        }
      }
    } catch (e) {}
  }, [captureFrame, lastAdded]);

  useEffect(() => {
    if (isRunning && mode === "sign2text") {
      intervalRef.current = setInterval(predict, CAPTURE_INTERVAL);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isRunning, mode, predict]);

  const speakSentence = async () => {
    const text = sentence.join(" ");
    if (!text) return;
    setSpeaking(true);
    try {
      const res = await fetch(`${API}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang })
      });
      const data = await res.json();
      const audio = new Audio(`data:audio/mp3;base64,${data.audio_base64}`);
      audio.onended = () => setSpeaking(false);
      audio.play();
    } catch {
      const utt = new SpeechSynthesisUtterance(text);
      utt.onend = () => setSpeaking(false);
      speechSynthesis.speak(utt);
    }
  };

  const wordsToSigns = textInput.trim().split(/\s+/).filter(Boolean);

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🤟</span>
          <span className="logo-text">BharatSign</span>
          <span className="logo-sub">Indian Sign Language AI</span>
        </div>
        <div className="mode-toggle">
          <button
            className={mode === "sign2text" ? "active" : ""}
            onClick={() => { setMode("sign2text"); setIsRunning(false); setSentence([]); }}
          >
            Sign → Text
          </button>
          <button
            className={mode === "text2sign" ? "active" : ""}
            onClick={() => { setMode("text2sign"); setIsRunning(false); }}
          >
            Text → Sign
          </button>
        </div>
        <select className="lang-select" value={lang} onChange={e => setLang(e.target.value)}>
          <option value="en">🇬🇧 English</option>
          <option value="hi">🇮🇳 Hindi</option>
          <option value="te">Telugu</option>
        </select>
      </header>

      <main className="main">
        {mode === "sign2text" ? (
          <div className="sign2text">
            <div className="video-container">
              <video ref={videoRef} autoPlay playsInline muted className="video" />
              <canvas ref={canvasRef} style={{ display: "none" }} />
              {!isRunning && (
                <div className="video-overlay">
                  <p>Click START to begin recognizing signs</p>
                </div>
              )}
              {isRunning && prediction.hand_detected && (
                <div className="hand-badge">✋ Hand Detected</div>
              )}
              {isRunning && !prediction.hand_detected && (
                <div className="hand-badge no-hand">No hand in frame</div>
              )}
            </div>

            <div className="prediction-panel">
              <div className="current-sign">
                {prediction.sign ? (
                  <>
                    <span className="sign-emoji">{SIGN_EMOJIS[prediction.sign] || "🤟"}</span>
                    <span className="sign-word">{prediction.sign}</span>
                    <span className="confidence">{(prediction.confidence * 100).toFixed(0)}% confident</span>
                    <div className="conf-bar">
                      <div className="conf-fill" style={{ width: `${prediction.confidence * 100}%` }} />
                    </div>
                  </>
                ) : (
                  <span className="idle-text">{isRunning ? "Show a sign..." : "Press START"}</span>
                )}
              </div>
              <button
                className={`start-btn ${isRunning ? "stop" : "start"}`}
                onClick={() => setIsRunning(r => !r)}
              >
                {isRunning ? "⏹ STOP" : "▶ START"}
              </button>
            </div>

            <div className="sentence-panel">
              <div className="sentence-header">
                <span>📝 Sentence Builder</span>
                <button className="clear-btn" onClick={() => setSentence([])}>Clear</button>
              </div>
              <div className="sentence-display">
                {sentence.length === 0
                  ? <span className="placeholder">Signs you hold for 1 second will appear here...</span>
                  : sentence.map((s, i) => (
                      <span key={i} className="word-chip">
                        {SIGN_EMOJIS[s] || "🤟"} {s}
                      </span>
                    ))
                }
              </div>
              <div className="sentence-actions">
                <button className="speak-btn" onClick={speakSentence} disabled={sentence.length === 0 || speaking}>
                  {speaking ? "🔊 Speaking..." : "🔊 Speak Sentence"}
                </button>
                <button className="undo-btn" onClick={() => setSentence(s => s.slice(0, -1))} disabled={sentence.length === 0}>
                  ↩ Undo
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="text2sign">
            <div className="text-input-panel">
              <h2>Type a word or sentence</h2>
              <input
                className="text-input"
                type="text"
                placeholder="e.g. Hello Thank You India"
                value={textInput}
                onChange={e => setTextInput(e.target.value)}
              />
              <p className="hint">Each word will show its ISL sign. Separate words with spaces.</p>
            </div>
            <div className="signs-grid">
              {wordsToSigns.length === 0
                ? signs.map(s => (
                    <div key={s} className="sign-card">
                      <div className="sign-card-emoji">{SIGN_EMOJIS[s] || "🤟"}</div>
                      <div className="sign-card-label">{s}</div>
                    </div>
                  ))
                : wordsToSigns.map((word, i) => {
                    const matched = signs.find(s => s.toLowerCase() === word.toLowerCase());
                    return (
                      <div key={i} className={`sign-card ${matched ? "matched" : "unknown"}`}>
                        <div className="sign-card-emoji">{matched ? (SIGN_EMOJIS[matched] || "🤟") : "❓"}</div>
                        <div className="sign-card-label">{word}</div>
                        {!matched && <div className="sign-card-sub">Not in ISL library</div>}
                      </div>
                    );
                  })
              }
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <span>🇮🇳 Built for 63 Lakh+ Deaf & Mute Indians</span>
        <span>HackIndia Spark-11 · Team XD · CBIT Hyderabad</span>
      </footer>
    </div>
  );
}