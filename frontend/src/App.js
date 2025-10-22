// App.jsx

import React, { useState } from "react";
import Navbar from "./navbar.js"
import InputSection from "./InputSection";


function App() {
  //State for storing backend result
  const [result, setResult] = useState(null);

  //State for loading to disable buttons while request is in progress
  const [loading, setLoading] = useState(false);
 

  // --- Upload Image handler ---
  const handleUpload = async (file) => {
    setLoading(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/upload-image", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error uploading image");
    } finally {
      setLoading(false);
    }
  };

  // --- Type Food Name handler ---
  const handleType = async (name) => {
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/get-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error sending food name");
    } finally {
      setLoading(false);
    }
  };

  // --- Take Picture handler ---
  const handleCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      await video.play();

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      setTimeout(() => {
        ctx.drawImage(video, 0, 0);
        canvas.toBlob(async (blob) => {
          await handleUpload(blob);
          stream.getTracks().forEach((track) => track.stop());
        }, "image/jpeg");
      }, 1000);
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

   // JSX: what the user sees on the page
  return (
    <div>  
        <Navbar />
       <InputSection 
        onUpload={handleUpload}
        onType={handleType}
        onCamera={handleCamera}
        loading={loading}
        result={result}
        />
  </div>
  );
}

export default App;

