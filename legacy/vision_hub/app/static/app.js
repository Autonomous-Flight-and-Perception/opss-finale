const $ = (id)=>document.getElementById(id);
const log = (t)=>{ const L=$("log"); L.textContent += t + "\n"; L.scrollTop=L.scrollHeight; };

async function start() {
  const r = await fetch("/api/api/camera/start", {method:"POST"});
  log("start: " + await r.text());
  await status();
}

async function stop() {
  const r = await fetch("/api/api/camera/stop", {method:"POST"});
  log("stop: " + await r.text());
  stopStreams();
  await status();
}

async function status() {
  const j = await (await fetch("/api/api/camera/status")).json();
  $("status").textContent = j.started ? "Running" : "Stopped";
  $("devinfo").textContent = "Local RealSense";
}

async function capture() {
  const r = await fetch("/api/capture/single", {method:"POST"});
  log("capture: " + await r.text());
}

function streamColor(){ 
  $("color").src = "/api/api/stream/color"; 
  log("Streaming from local RealSense with YOLOv8");
}

function streamDepth(){ 
  $("depth").src = "/api/api/stream/depth"; 
}

function stopStreams(){ 
  $("color").src=""; 
  $("depth").src=""; 
}

$("start").onclick = start;
$("stop").onclick = stop;
$("check").onclick = status;
$("capture").onclick = capture;
$("streamColor").onclick = streamColor;
$("streamDepth").onclick = streamDepth;
$("stopStreams").onclick = stopStreams;

status();
