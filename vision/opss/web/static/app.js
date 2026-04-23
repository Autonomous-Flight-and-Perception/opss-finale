const $ = (id) => document.getElementById(id);
const log = (t) => { const L = $("log"); L.textContent += t + "\n"; L.scrollTop = L.scrollHeight; };

async function start() {
  const r = await fetch("/pipeline/start", { method: "POST" });
  log("start: " + await r.text());
  await status();
}

async function stop() {
  const r = await fetch("/pipeline/stop", { method: "POST" });
  log("stop: " + await r.text());
  stopStreams();
  await status();
}

async function status() {
  const j = await (await fetch("/pipeline/status")).json();
  $("status").textContent = j.running ? "Running" : "Stopped";
  $("devinfo").textContent = j.config ? j.config.capture_resolution : "—";
}

function streamColor() {
  $("color").src = "/stream/color";
  log("Streaming annotated color from host-side OPSS (YOLOv8)");
}

function stopStreams() {
  $("color").src = "";
}

$("start").onclick = start;
$("stop").onclick = stop;
$("check").onclick = status;
$("streamColor").onclick = streamColor;
$("stopStreams").onclick = stopStreams;

status();
