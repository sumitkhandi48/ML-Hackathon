/**
 * PathSnap Trace-Lite Client Application
 * Communicates with FastAPI backend over HTTP / LAN IP.
 * Renders interactive Leaflet map layers, performance metrics, and track warnings.
 */

// State
let map = null;
let gpsLayerGroup = null;
let routeLayerGroup = null;
let roadsLayerGroup = null;
let markersLayerGroup = null;

let currentBounds = null;

// DOM Elements
const serverUrlInput = document.getElementById("serverUrl");
const connectBtn = document.getElementById("connectBtn");
const tripSelect = document.getElementById("tripSelect");
const runBtn = document.getElementById("runBtn");
const useHmmToggle = document.getElementById("useHmmToggle");
const connectionStatus = document.getElementById("connectionStatus");
const connectionText = document.getElementById("connectionText");

const valTop1 = document.getElementById("valTop1");
const valContinuity = document.getElementById("valContinuity");
const valLcs = document.getElementById("valLcs");
const valLatency = document.getElementById("valLatency");

const diagnosticsInfo = document.getElementById("diagnosticsInfo");
const warningsList = document.getElementById("warningsList");
const terminalLogs = document.getElementById("terminalLogs");
const clearLogsBtn = document.getElementById("clearLogsBtn");

const layerGps = document.getElementById("layerGps");
const layerRoute = document.getElementById("layerRoute");
const layerRoads = document.getElementById("layerRoads");
const fitBoundsBtn = document.getElementById("fitBoundsBtn");
const clearMapBtn = document.getElementById("clearMapBtn");

// Initialize Map
function initMap() {
  map = L.map("map", {
    center: [49.8728, 8.6512],
    zoom: 13,
    zoomControl: true
  });

  // Base tile layer with fallback error handling
  const tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors"
  });
  tileLayer.addTo(map);

  // Initialize Feature Groups
  roadsLayerGroup = L.featureGroup().addTo(map);
  routeLayerGroup = L.featureGroup().addTo(map);
  gpsLayerGroup = L.featureGroup().addTo(map);
  markersLayerGroup = L.featureGroup().addTo(map);

  // Wire layer visibility checkboxes
  layerGps.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(gpsLayerGroup);
    else map.removeLayer(gpsLayerGroup);
  });

  layerRoute.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(routeLayerGroup);
    else map.removeLayer(routeLayerGroup);
  });

  layerRoads.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(roadsLayerGroup);
    else map.removeLayer(roadsLayerGroup);
  });

  fitBoundsBtn.addEventListener("click", () => {
    if (currentBounds) map.fitBounds(currentBounds, { padding: [20, 20] });
  });

  clearMapBtn.addEventListener("click", clearMap);
}

function logTerminal(msg, level = "info") {
  const line = document.createElement("div");
  line.className = `log-line ${level}`;
  const timestamp = new Date().toLocaleTimeString();
  line.textContent = `[${timestamp}] ${msg}`;
  terminalLogs.appendChild(line);
  terminalLogs.scrollTop = terminalLogs.scrollHeight;
}

clearLogsBtn.addEventListener("click", () => {
  terminalLogs.innerHTML = "";
});

function getBaseUrl() {
  let url = serverUrlInput.value.trim();
  if (url.endsWith("/")) url = url.slice(0, -1);
  return url;
}

// Check Backend Health & Fetch Trips
async function checkHealthAndLoadTrips() {
  const baseUrl = getBaseUrl();
  logTerminal(`Checking connection to backend at ${baseUrl}...`, "info");
  
  try {
    const healthRes = await fetch(`${baseUrl}/health`);
    if (!healthRes.ok) throw new Error(`HTTP ${healthRes.status}`);
    const healthData = await healthRes.json();
    
    connectionStatus.querySelector(".status-indicator").className = "status-indicator online";
    connectionText.textContent = `Connected (${healthData.model_loaded ? "Model Ready" : "Loading Model..."})`;
    logTerminal(`Backend online. Device: ${healthData.device}. Model loaded: ${healthData.model_loaded}`, "success");

    // Fetch Trips
    logTerminal(`Fetching available test trips...`, "info");
    const tripsRes = await fetch(`${baseUrl}/trips?partition=student_test`);
    if (!tripsRes.ok) throw new Error(`HTTP ${tripsRes.status}`);
    const tripsData = await tripsRes.json();

    tripSelect.innerHTML = "";
    tripsData.trips.forEach((tid, idx) => {
      const opt = document.createElement("option");
      opt.value = tid;
      opt.textContent = `Trip ${tid} (Test #${idx + 1})`;
      tripSelect.appendChild(opt);
    });

    runBtn.disabled = false;
    logTerminal(`Discovered ${tripsData.count} evaluator trips in partition '${tripsData.partition}'.`, "success");
  } catch (err) {
    connectionStatus.querySelector(".status-indicator").className = "status-indicator offline";
    connectionText.textContent = "Offline / Error";
    logTerminal(`Connection failed: ${err.message}. Check that FastAPI is running on host.`, "error");
    runBtn.disabled = true;
  }
}

connectBtn.addEventListener("click", checkHealthAndLoadTrips);

function clearMap() {
  gpsLayerGroup.clearLayers();
  routeLayerGroup.clearLayers();
  roadsLayerGroup.clearLayers();
  markersLayerGroup.clearLayers();
  currentBounds = null;
}

// Run Prediction Request
async function runPrediction() {
  const tripId = tripSelect.value;
  if (!tripId) return;

  const baseUrl = getBaseUrl();
  const useHmm = useHmmToggle.checked;

  runBtn.disabled = true;
  document.getElementById("btnText").textContent = "Processing...";
  logTerminal(`Dispatching POST /predict for trip ${tripId} (HMM=${useHmm})...`, "info");

  try {
    const res = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trip_id: tripId,
        partition: "student_test",
        use_hmm: useHmm
      })
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(errJson.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    logTerminal(`Inference completed in ${data.inference_time_ms} ms. GPS observations: ${data.num_observations}.`, "success");

    // 1. Update Metrics Cards
    if (data.metrics && data.metrics.evaluation_available) {
      valTop1.textContent = `${data.metrics.top1_accuracy_percent}%`;
      valContinuity.textContent = `${data.metrics.route_continuity_percent}%`;
      valLcs.textContent = `${data.metrics.sequence_lcs_f1_percent}%`;
      logTerminal(`Server-side Evaluation: Top-1 Accuracy: ${data.metrics.top1_accuracy_percent}%, Route Continuity: ${data.metrics.route_continuity_percent}%, Sequence LCS F1: ${data.metrics.sequence_lcs_f1_percent}%.`, "success");
    } else {
      valTop1.textContent = "N/A";
      valContinuity.textContent = "N/A";
      valLcs.textContent = "N/A";
    }
    valLatency.textContent = `${data.inference_time_ms} ms`;

    // 2. Diagnostics & Warnings
    diagnosticsInfo.innerHTML = `
      <p style="font-size: 12px; margin-bottom: 6px;"><strong>Trip ID:</strong> ${data.trip_id} | <strong>Observations:</strong> ${data.num_observations}</p>
      <p style="font-size: 12px;"><strong>Reconstructed Arcs:</strong> ${data.predicted_route_segments.length} segments</p>
    `;

    warningsList.innerHTML = "";
    if (data.warnings && data.warnings.length > 0) {
      logTerminal(`Warning: ${data.warnings.length} irregular timestamp gaps (>5s) detected in trajectory.`, "warn");
      data.warnings.forEach(w => {
        const item = document.createElement("div");
        item.className = "warning-item";
        item.textContent = `Obs ${w.obs_index}: Δt = ${w.gap_seconds.toFixed(1)}s (gap at t=${w.timestamp}s)`;
        warningsList.appendChild(item);
      });
    } else {
      const item = document.createElement("div");
      item.className = "empty-state";
      item.textContent = "No large timestamp gaps detected (uniform 1Hz sampling).";
      warningsList.appendChild(item);
    }

    // 3. Render Map Layers
    clearMap();

    // Road network layer (subsampled background)
    if (data.road_network_segments) {
      data.road_network_segments.forEach(seg => {
        L.polyline(seg, { color: "#64748b", weight: 1.5, opacity: 0.5 }).addTo(roadsLayerGroup);
      });
    }

    // Predicted route layer (bold blue)
    if (data.predicted_route_segments && data.predicted_route_segments.length > 0) {
      data.predicted_route_segments.forEach(seg => {
        L.polyline(seg, { color: "#3b82f6", weight: 4.5, opacity: 0.9 }).addTo(routeLayerGroup);
      });
    }

    // GPS points layer (red circle markers)
    const latLngs = [];
    if (data.gps_points) {
      data.gps_points.forEach((pt, idx) => {
        latLngs.push(pt);
        // Add marker every 2 points for performance
        if (idx % 2 === 0) {
          L.circleMarker(pt, {
            radius: 3.5,
            color: "#ef4444",
            fillColor: "#ef4444",
            fillOpacity: 0.8,
            weight: 1
          }).bindPopup(`GPS Obs #${idx}: [${pt[0].toFixed(5)}, ${pt[1].toFixed(5)}]`).addTo(gpsLayerGroup);
        }
      });
    }

    // Start & End Markers
    if (latLngs.length > 0) {
      const startPt = latLngs[0];
      const endPt = latLngs[latLngs.length - 1];

      L.circleMarker(startPt, {
        radius: 7,
        color: "#10b981",
        fillColor: "#10b981",
        fillOpacity: 1
      }).bindPopup(`<strong>Trajectory Start</strong><br>Obs 0`).addTo(markersLayerGroup);

      L.circleMarker(endPt, {
        radius: 7,
        color: "#8b5cf6",
        fillColor: "#8b5cf6",
        fillOpacity: 1
      }).bindPopup(`<strong>Trajectory End</strong><br>Obs ${latLngs.length - 1}`).addTo(markersLayerGroup);

      currentBounds = L.latLngBounds(latLngs);
      map.fitBounds(currentBounds, { padding: [30, 30] });
    }

  } catch (err) {
    logTerminal(`Inference failed: ${err.message}`, "error");
    alert(`Inference Error: ${err.message}`);
  } finally {
    runBtn.disabled = false;
    document.getElementById("btnText").textContent = "Run Map-Matching";
  }
}

runBtn.addEventListener("click", runPrediction);

// Auto-init on load
window.addEventListener("DOMContentLoaded", () => {
  initMap();
  checkHealthAndLoadTrips();
});
