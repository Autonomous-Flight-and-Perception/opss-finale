const $ = (id)=>document.getElementById(id);
const log = (t)=>{ const L=$("log"); L.textContent += t + "\n"; L.scrollTop=L.scrollHeight; };

async function start() {
  const r = await fetch("/camera/start", {method:"POST"});
  log("start: " + await r.text());
  await status();
}

async function stop() {
  const r = await fetch("/camera/stop", {method:"POST"});
  log("stop: " + await r.text());
  stopStreams();
  await status();
}

async function status() {
  const j = await (await fetch("/camera/status")).json();
  $("status").textContent = j.started ? "Running" : "Stopped";
  $("devinfo").textContent = "Local RealSense";
}

async function capture() {
  const r = await fetch("/capture/single", {method:"POST"});
  log("capture: " + await r.text());
}

function streamColor(){ 
  $("color").src = "/stream/color"; 
  log("Streaming from local RealSense with YOLOv8");
}

function streamDepth(){ 
  $("depth").src = "/stream/depth"; 
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
