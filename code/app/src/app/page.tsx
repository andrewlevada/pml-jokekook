"use client";

import { useState } from "react";

export default function Home() {
  const [joke, setJoke] = useState("");
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = () => {
    setLoading(true);
    const endpoint = "https://andrewlevada--generate.modal.run";

    fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({ topic }),
    })
      .then((response) => response.json())
      .then((data) => setJoke(data.joke))
      .catch((error) => console.error("Error:", error))
      .finally(() => setLoading(false));
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen w-screen p-[200px]">
      <div className="flex flex-col gap-4 w-full max-w-md">
        <label htmlFor="topic" className="text-lg font-bold">A joke about</label>
        <input
          type="text"
          placeholder="cats"
          className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
        
        <button className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors" onClick={handleSubmit}>
          {loading ? "Thinking..." : "Think of a joke"}
        </button>

        <div className="w-full p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          {joke && (
            <p className="text-gray-700 dark:text-gray-300">
              {joke}
            </p>
          )}
          {!joke && (
            <p className="text-gray-500 dark:text-gray-500">
              Get ready to laugh
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
