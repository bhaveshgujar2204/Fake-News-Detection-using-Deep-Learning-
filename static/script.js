async function checkNews() {
    const newsTitle = document.getElementById("newsTitle").value;
    
    console.log("🔍 Sending request with:", newsTitle); // Debugging step

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newsTitle })
    });

    const data = await response.json();
    console.log("📩 Received response:", data); // Debugging step

    document.getElementById("result").innerText = data.prediction || "Error: No response";
}
