import * as drawingUtils from "@mediapipe/drawing_utils";
import type { Results } from "@mediapipe/hands";
import { HAND_CONNECTIONS, Hands } from "@mediapipe/hands";
import { useEffect, useRef, useState } from 'react';
import './App.css';
import { sendLandmarksToBackend } from "./utils/predict";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<(Hands & { _ready?: boolean }) | null>(null);
  const lastPredictionTimeRef = useRef<number>(0);
  const predictionCooldown = 2000;
  const animationFrameIdRef = useRef<number | null>(null);

  const [mode, setMode] = useState<"default" | "practice" | "test" | "home">("home");
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState(false);
  const [targetLetter, setTargetLetter] = useState("A");
  const [paused, setPaused] = useState(false);

  const [testRound, setTestRound] = useState(1);
  const [score, setScore] = useState(0);
  const [countdown, setCountdown] = useState(10);
  const [showTestResult, setShowTestResult] = useState(false);
  const [wasCorrectThisRound, setWasCorrectThisRound] = useState(false);
  const [backgroundColor, setBackgroundColor] = useState("bg-yellow-200");


  const targetLetterRef = useRef(targetLetter);
  const wasCorrectRef = useRef(wasCorrectThisRound);

  useEffect(() => { targetLetterRef.current = targetLetter; }, [targetLetter]);
  useEffect(() => { wasCorrectRef.current = wasCorrectThisRound; }, [wasCorrectThisRound]);

  function advanceTestRound() {
    if (testRound >= 10) {
      setShowTestResult(true);
    } else {
      setTestRound(prev => prev + 1);
      setCountdown(10);
      setWasCorrectThisRound(false);
      setTargetLetter(getRandomLetter());
    }
  }

  function resetHome(){
    const video = videoRef.current;
    const stream = video?.srcObject as MediaStream;
    stream?.getTracks().forEach(track => track.stop());
    if (video) video.srcObject = null;

    // Reset all states
    setMode("home");
    setScore(0);
    setTestRound(1);
    setCountdown(10);
    setPrediction(null);
    setPaused(false);
    setIsCorrect(false);
    setShowTestResult(false);
    setWasCorrectThisRound(false);
    setTargetLetter("A"); // or getRandomLetter()
    
    // Optionally reset MediaPipe (you already do this in cleanup)
    handsRef.current = null;
  }

  const letters = "ABCDEFGHIKLMNOPQRSTUVWXY";
  
  function getRandomLetter(): string {
    return letters[Math.floor(Math.random() * letters.length)];
  }

  const startWebcam = async () => {
    const video = videoRef.current;
    if (!video) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      video.srcObject = stream;
      await video.play();

      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const onFrame = async () => {
        if (paused || !handsRef.current || !handsRef.current._ready) return;
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          await handsRef.current.send({ image: video });
        }
          animationFrameIdRef.current = requestAnimationFrame(onFrame);
      };

      animationFrameIdRef.current = requestAnimationFrame(onFrame);
    } catch (err) {
      console.error("Webcam error:", err);
    }
  };

  useEffect(() => {
  if (isCorrect || wasCorrectThisRound) {
    setBackgroundColor("bg-green-300");
  } else if (prediction && prediction !== targetLetter) {
    setBackgroundColor("bg-red-300");
  } else {
    setBackgroundColor("bg-yellow-200");
  }
}, [isCorrect, wasCorrectThisRound, prediction, targetLetter]);


  useEffect(() => {
    if (mode !== "test" || showTestResult) return;
    if (countdown === 0) {
      advanceTestRound();
      return;
    }
    const timer = setTimeout(() => {
      setCountdown(prev => prev - 1);
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown, mode, showTestResult]);

   useEffect(() => {
    if (mode === "practice" && prediction === targetLetter && !paused) {
      
      setPaused(true);
      setIsCorrect(true);
    }
  }, [prediction, mode, targetLetter, paused]);
  
  useEffect(() => {
    if(mode === "practice" || mode === "test"){
      setScore(0);
      setTestRound(1);
      setCountdown(10);
      setWasCorrectThisRound(false);
      setPrediction(null);
      setTargetLetter(getRandomLetter());
      setPaused(false);
      setIsCorrect(false);
    }
  }, [mode])

 

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!video || !canvas || !ctx) return;

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    }) as Hands & { _ready?: boolean };

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults(async (results: Results) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
          drawingUtils.drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
            color: "blue", lineWidth: 3
          });
          drawingUtils.drawLandmarks(ctx, landmarks, {
            color: "red", radius: 3
          });
        }

        const now = Date.now();
        if (!paused && now - lastPredictionTimeRef.current > predictionCooldown) {
          const rawLandmarks = results.multiHandLandmarks[0];
          if (rawLandmarks && rawLandmarks.length === 21) {
            const letter = await sendLandmarksToBackend(rawLandmarks);
            if (letter) {
              setPrediction(letter);
              if (mode === "test" && !wasCorrectRef.current && letter === targetLetterRef.current) {
                setScore(prev => prev + 1);
                setWasCorrectThisRound(true);
              }
              if (mode === "practice" && letter === targetLetter) {
                setPaused(true);
                setIsCorrect(true);
              }
            }
            lastPredictionTimeRef.current = now;
          }
        }
      }
    });

    hands._ready = true;
    handsRef.current = hands;
    if (mode === "practice" || mode === "test") startWebcam();

    return () => {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
        animationFrameIdRef.current = null;
      }

      hands.close();
      handsRef.current = null;
      const stream = video.srcObject as MediaStream;
      stream?.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    };
  }, [mode]);



  return (
    <div className={`relative w-full h-screen bg-zinc-600`}>

      {mode === "home" && (
        <div className = "flex items-center justify-center h-screen w-full bg-gradient-to-r from-purple-500 via-purple-400 to-fuchsia-500">
        <div className="flex flex-col gap-6">
        <h1 className = "text-6xl mb-4 font-extrabold "  >ASL Vision</h1>
        <h1 className = "text-3xl mb-4 font-extrabold  "  > A simple ASL Alphabet Learning Application </h1>
        <button 
          onClick={() => setMode("practice")} className={`min-w-[600px] self-center px-5 py-2 rounded-md z-50  bg-green-500 text-white ` }>Practice Mode</button>
        <button 
          onClick={() => setMode("test")} className={`min-w-[600px] self-center px-5 py-2 rounded-md z-50 bg-red-500 text-white`}>Test Mode</button>
          </div>
          </div>
      )}



      {/* Mode Buttons */}
      {(mode === "practice" || mode === "test") && (
        <div>
          <button 
            onClick={() => setMode("practice")} 
            className={`absolute top-4 left-4 mb-4 min-w-[200px] px-5 py-2 rounded-md z-50 ${mode === "practice" ? "bg-green-500 text-white" : "bg-red-500 text-black"}` }>Practice Mode</button>
          <button 
            onClick={() => setMode("test")} 
            className={`absolute top-20 left-4 mb-4 min-w-[200px] px-5 py-2 rounded-md z-50 ${mode === "test" ? "bg-green-500 text-white" : "bg-red-500 text-white"}`}>Test Mode</button>
          <button 
            onClick={() => resetHome()} 
            className={`absolute top-36 left-4 mb-4 min-w-[200px] px-5 py-2 rounded-md z-50 bg-blue-500 text-white`}>Return to home</button>
         
          </div>
        
      )}
      
      {/* UI per mode */}
      {mode === "practice" && (
      <div className={`flex flex-col items-center justify-start w-full h-screen bg-zinc-600 pt-12`}>

        {/* Title */}
        <h1 className="text-3xl font-bold font-mono mb-6 text-zinc-300">
          ASLVision: Simple ASL Alphabet Learner
        </h1>

        {/* Framed Box Containing Sign + Webcam */}
        <div className={`flex gap-8 border-4 border-black rounded-xl p-6 shadow-lg ${backgroundColor}`}>

          {/* Left: Sign */}
          <div className="flex flex-col items-center justify-center">
            <img src={`/signs/${targetLetter}.png`} alt={targetLetter} className="w-[300px] h-auto" />
          </div>

          {/* Right: Webcam */}
          <div className="relative">
            <video ref={videoRef} className="rounded-lg w-[800px] h-[600px] z-10 transform scaleX(-1)" autoPlay playsInline />
            <canvas ref={canvasRef} className="absolute top-0 left-0 w-[800px] h-[600px] z-20 pointer-events-none" width={640} height={480} />
          </div>
        </div>

        {/* Feedback Message */}
        {(isCorrect || (prediction && prediction !== targetLetter && !isCorrect)) && (
          <div className={`mt-4 px-6 py-3 rounded-md text-white font-semibold text-lg shadow-md
            ${isCorrect ? "bg-green-600" : "bg-red-600"}`}>
            {isCorrect ? "Correct!" : `Try Again! You signed ${prediction}, not ${targetLetter}`}
          </div>
        )}

        {/* Practice Button */}
        <button
          onClick={() => {
            setTargetLetter(getRandomLetter());
            setPrediction(null);
            setPaused(false);
            setIsCorrect(false);
          }}
          className="mt-6 px-5 py-2 rounded-md bg-blue-600 text-white font-semibold"
        >
          Practice Next Letter
        </button>
      </div>
    )}


      {mode === "test" && (
  <div className={`flex flex-col items-center justify-start w-full h-screen bg-zinc-600 pt-12`}>

    {/* Title */}
    <h1 className="text-3xl font-bold font-mono mb-6 text-zinc-300">
      ASLVision: Simple ASL Alphabet Learner
    </h1>

    {!showTestResult ? (
      <>
        {/* Framed Sign + Webcam */}
        <div className={`flex gap-8 border-4 border-black rounded-xl p-6 shadow-lg ${backgroundColor}`}>

          {/* Left: Sign */}
          <div className="flex flex-col items-center justify-center">
            <h2 className="text-2xl font-bold mb-2">Round {testRound}/10</h2>
            <p className="text-lg mb-2">Sign the letter:</p>
            <img src={`/signs/${targetLetter}.png`} alt={targetLetter} className="w-[300px] h-auto mb-2" />
            <p className="text-lg font-mono">Time left: {countdown}s</p>
          </div>

          {/* Right: Webcam */}
          <div className="relative">
            <video ref={videoRef} className="rounded-lg w-[800px] h-[600px] z-10 transform scaleX(-1)" autoPlay playsInline />
            <canvas ref={canvasRef} className="absolute top-0 left-0 w-[800px] h-[600px] z-20 pointer-events-none" width={640} height={480} />
          </div>
        </div>

        {/* Feedback Message */}
        {wasCorrectThisRound && (
          <div className="mt-4 px-6 py-3 rounded-md bg-green-600 text-white font-semibold text-lg shadow-md">
            Correct!
          </div>
        )}
      </>
    ) : (
      <>
        {/* Test Result */}
        <h2 className="text-3xl font-bold mb-6">Test Complete!</h2>
        <p className="text-xl mb-4">Your Score: {score}/10</p>
        <button
          onClick={() => {
            setMode("home");
            resetHome();
          }}
          className="px-6 py-2 rounded-md bg-blue-600 text-white font-semibold"
        >
          Try Again
        </button>
      </>
    )}
  </div>
)}

    </div>
  );
}

export default App;
