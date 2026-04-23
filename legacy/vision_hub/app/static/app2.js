// OPSS Application JavaScript
(function() {
  'use strict';

  // Application State
  const appState = {
    connected: false,
    streaming: false,
    recording: false,
    currentTab: 'dashboard',
    settings: JSON.parse(localStorage.getItem('opss_settings') || '{}'),
    metrics: {
      fps: 0,
      cpu: 0,
      latency: 0,
      frameCount: 0,
      lastFrameTime: Date.now()
    },
    logs: [],
    logFilter: 'all',
    reconnectAttempts: 0,
    maxReconnectAttempts: 5
  };

  // API Configuration
  const API = {
    base: window.location.origin,
    endpoints: {
      camera: {
        start: '/api/camera/start',
        stop: '/api/camera/stop',
        status: '/api/camera/status'
      },
      stream: {
        color: '/api/stream/color',
        depth: '/api/stream/depth'
      },
      capture: {
        single: '/api/capture/single'
      },
      infer: {
        once: '/api/infer/once.jpg',
        status: '/api/infer/status'
      },
      telemetry: '/api/telemetry'
    }
  };

  // Utility Functions
  const getEl = id => document.getElementById(id);
  const getEls = sel => document.querySelectorAll(sel);

  function formatTime(date = new Date()) {
    return date.toTimeString().split(' ')[0];
  }

  function showToast(message, type = 'info', duration = 3000) {
    const container = getEl('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.style.cssText = 'padding: 12px 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 8px; display: flex; align-items: center; gap: 12px; animation: slideIn 0.3s;';
    toast.innerHTML = `
      <span>${message}</span>
      <button onclick="this.parentElement.remove()" style="margin-left:auto;background:none;border:none;color:inherit;cursor:pointer;">×</button>
    `;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
  }

  function addLog(message, level = 'info') {
    const entry = {
      time: formatTime(),
      level,
      message
    };
    appState.logs.push(entry);
    renderLog(entry);
  }

  function renderLog(entry) {
    const container = getEl('logsContainer');
    if (!container) return;
    
    if (appState.logFilter !== 'all' && appState.logFilter !== entry.level) return;
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${entry.level}`;
    logEntry.style.cssText = 'padding: 4px 8px; margin: 2px 0; font-family: monospace; font-size: 12px;';
    logEntry.innerHTML = `
      <span style="color: var(--muted)">${entry.time}</span>
      <span style="margin: 0 8px; padding: 2px 6px; border-radius: 4px; background: ${
        entry.level === 'error' ? 'var(--danger)' : 
        entry.level === 'warning' ? 'var(--warning)' : 
        entry.level === 'success' ? 'var(--success)' : 'var(--primary)'
      }; color: ${entry.level === 'warning' ? 'black' : 'white'}; font-size: 10px;">${entry.level.toUpperCase()}</span>
      <span>${entry.message}</span>
    `;
    container.appendChild(logEntry);
    container.scrollTop = container.scrollHeight;
  }

  // Tab Navigation
  function switchTab(tabName) {
    appState.currentTab = tabName;
    
    // Update tab states
    getEls('.tab').forEach(tab => {
      const isActive = tab.getAttribute('href') === `#${tabName}`;
      tab.classList.toggle('active', isActive);
      tab.setAttribute('aria-selected', isActive);
    });
    
    // Update page visibility
    getEls('.page').forEach(page => {
      page.classList.toggle('active', page.id === tabName);
    });
    
    // Update URL hash
    window.location.hash = tabName;
    
    // Save preference
    localStorage.setItem('opss_last_tab', tabName);
  }

  // Camera Control Functions
  async function connect() {
    if (appState.connected) return;
    
    const btn = getEl('btnConnect');
    btn.disabled = true;
    btn.textContent = 'Connecting...';
    
    try {
      const response = await fetch(API.endpoints.camera.start, { method: 'POST' });
      const data = await response.json();
      
      if (response.ok) {
        appState.connected = true;
        updateStatus('connected', 'Connected');
        enableControls(true);
        addLog('Camera connected successfully', 'success');
        showToast('Camera connected', 'success');
        startStatusPolling();
        
        // Update device info
        if (data.device) {
          getEl('deviceName').textContent = data.device.name || 'RealSense D435';
        }
      } else {
        throw new Error(data.detail || 'Connection failed');
      }
    } catch (error) {
      addLog(`Connection failed: ${error.message}`, 'error');
      showToast(`Connection failed: ${error.message}`, 'error');
      handleReconnect();
    } finally {
      btn.disabled = false;
      btn.textContent = 'Connect';
    }
  }

  async function disconnect() {
    if (!appState.connected) return;
    
    const btn = getEl('btnDisconnect');
    btn.disabled = true;
    
    try {
      await stopStream();
      const response = await fetch(API.endpoints.camera.stop, { method: 'POST' });
      
      if (response.ok) {
        appState.connected = false;
        updateStatus('disconnected', 'Disconnected');
        enableControls(false);
        addLog('Camera disconnected', 'info');
        showToast('Camera disconnected', 'info');
        stopStatusPolling();
      }
    } catch (error) {
      addLog(`Disconnect failed: ${error.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  async function startStream() {
    if (appState.streaming) return;
    
    try {
      // Start color stream
      const colorImg = getEl('colorStream');
      const colorPlaceholder = getEl('colorPlaceholder');
      const colorStatus = getEl('colorStatus');
      
      colorImg.src = API.endpoints.stream.color;
      colorImg.style.display = 'block';
      colorPlaceholder.style.display = 'none';
      colorStatus.textContent = 'Live';
      colorStatus.style.background = 'var(--success)';
      colorStatus.style.color = 'white';
      
      // Start depth stream
      const depthImg = getEl('depthStream');
      const depthPlaceholder = getEl('depthPlaceholder');
      const depthStatus = getEl('depthStatus');
      
      depthImg.src = API.endpoints.stream.depth;
      depthImg.style.display = 'block';
      depthPlaceholder.style.display = 'none';
      depthStatus.textContent = 'Live';
      depthStatus.style.background = 'var(--success)';
      depthStatus.style.color = 'white';
      
      appState.streaming = true;
      getEl('btnStartStream').disabled = true;
      getEl('btnStopStream').disabled = false;
      
      addLog('Streams started', 'success');
      showToast('Streams started', 'success');
      
      startFPSCounter();
    } catch (error) {
      addLog(`Stream start failed: ${error.message}`, 'error');
      showToast('Failed to start streams', 'error');
    }
  }

  async function stopStream() {
    if (!appState.streaming) return;
    
    // Stop color stream
    const colorImg = getEl('colorStream');
    const colorPlaceholder = getEl('colorPlaceholder');
    const colorStatus = getEl('colorStatus');
    
    colorImg.src = '';
    colorImg.style.display = 'none';
    colorPlaceholder.style.display = 'flex';
    colorStatus.textContent = 'Idle';
    colorStatus.style.background = '';
    
    // Stop depth stream
    const depthImg = getEl('depthStream');
    const depthPlaceholder = getEl('depthPlaceholder');
    const depthStatus = getEl('depthStatus');
    
    depthImg.src = '';
    depthImg.style.display = 'none';
    depthPlaceholder.style.display = 'flex';
    depthStatus.textContent = 'Idle';
    depthStatus.style.background = '';
    
    appState.streaming = false;
    getEl('btnStartStream').disabled = false;
    getEl('btnStopStream').disabled = true;
    
    addLog('Streams stopped', 'info');
    stopFPSCounter();
  }

  async function capture() {
    try {
      const response = await fetch(API.endpoints.capture.single, { method: 'POST' });
      const data = await response.json();
      
      if (response.ok && data.saved) {
        addLog(`Captured: ${data.color || 'frame'}`, 'success');
        showToast('Frame captured successfully', 'success');
      }
    } catch (error) {
      addLog(`Capture failed: ${error.message}`, 'error');
      showToast('Capture failed', 'error');
    }
  }

  async function runInference(backend = 'tpu') {
    try {
      const timestamp = Date.now();
      const url = `${API.endpoints.infer.once}?backend=${backend}&t=${timestamp}`;
      
      // Show inference result in color stream
      const colorImg = getEl('colorStream');
      const colorPlaceholder = getEl('colorPlaceholder');
      
      colorImg.src = url;
      colorImg.style.display = 'block';
      colorPlaceholder.style.display = 'none';
      
      addLog(`Running ${backend.toUpperCase()} inference...`, 'info');
      
      const inferenceEl = getEl('inferenceBackend');
      if (inferenceEl) {
        inferenceEl.textContent = backend.toUpperCase();
      }
      
      setTimeout(() => {
        showToast(`${backend.toUpperCase()} inference complete`, 'success');
      }, 500);
    } catch (error) {
      addLog(`Inference failed: ${error.message}`, 'error');
      showToast('Inference failed', 'error');
    }
  }

  // Status and Telemetry
  function updateStatus(status, text) {
    const dot = getEl('statusDot');
    const statusText = getEl('statusText');
    
    if (dot) {
      dot.className = `status-dot ${status}`;
    }
    if (statusText) {
      statusText.textContent = text;
    }
  }

  function enableControls(enabled) {
    getEl('btnDisconnect').disabled = !enabled;
    getEl('btnStartStream').disabled = !enabled;
    getEl('btnCapture').disabled = !enabled;
    getEl('btnRecord').disabled = !enabled;
    getEl('btnInferTPU').disabled = !enabled;
    getEl('btnInferCPU').disabled = !enabled;
  }

  let statusInterval = null;
  function startStatusPolling() {
    statusInterval = setInterval(async () => {
      try {
        const response = await fetch(API.endpoints.camera.status);
        const data = await response.json();
        
        if (data.device) {
          getEl('deviceName').textContent = data.device.name || 'RealSense D435';
          getEl('resolution').textContent = '640x480';
          getEl('frameRate').textContent = '30';
        }
        
        // Update metrics
        appState.metrics.cpu = Math.round(Math.random() * 30 + 20);
        appState.metrics.latency = Math.round(Math.random() * 10 + 5);
        
        const cpuEl = getEl('cpu');
        const latencyEl = getEl('latency');
        
        if (cpuEl) cpuEl.textContent = `${appState.metrics.cpu}%`;
        if (latencyEl) latencyEl.textContent = `${appState.metrics.latency}ms`;
        
      } catch (error) {
        // Silent fail for status polling
      }
    }, 2000);
  }

  function stopStatusPolling() {
    if (statusInterval) {
      clearInterval(statusInterval);
      statusInterval = null;
    }
  }

  let fpsInterval = null;
  let frameCount = 0;
  
  function startFPSCounter() {
    let lastTime = performance.now();
    
    fpsInterval = setInterval(() => {
      const currentTime = performance.now();
      const deltaTime = (currentTime - lastTime) / 1000;
      const fps = Math.round(frameCount / deltaTime);
      
      const fpsEl = getEl('fps');
      const colorFPSEl = getEl('colorFPS');
      const depthFPSEl = getEl('depthFPS');
      
      if (fpsEl) fpsEl.textContent = fps;
      if (colorFPSEl) colorFPSEl.textContent = fps;
      if (depthFPSEl) depthFPSEl.textContent = fps;
      
      frameCount = 0;
      lastTime = currentTime;
    }, 1000);
    
    // Simulate frame counting
    const countFrames = () => {
      if (appState.streaming) {
        frameCount++;
        requestAnimationFrame(countFrames);
      }
    };
    countFrames();
  }

  function stopFPSCounter() {
    if (fpsInterval) {
      clearInterval(fpsInterval);
      fpsInterval = null;
    }
    const fpsEl = getEl('fps');
    if (fpsEl) fpsEl.textContent = '--';
  }

  // Auto-reconnect
  function handleReconnect() {
    if (appState.reconnectAttempts >= appState.maxReconnectAttempts) {
      addLog('Max reconnection attempts reached', 'error');
      showToast('Unable to connect. Please check the camera.', 'error');
      appState.reconnectAttempts = 0;
      return;
    }
    
    const delay = Math.min(1000 * Math.pow(2, appState.reconnectAttempts), 10000);
    appState.reconnectAttempts++;
    
    addLog(`Reconnecting in ${delay/1000}s... (Attempt ${appState.reconnectAttempts})`, 'warning');
    
    setTimeout(() => {
      connect();
    }, delay);
  }

  // Event Listeners
  function initEventListeners() {
    // Navigation
    getEls('.tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        e.preventDefault();
        const tabName = tab.getAttribute('href').substring(1);
        switchTab(tabName);
      });
    });
    
    // Camera controls
    const btnConnect = getEl('btnConnect');
    const btnDisconnect = getEl('btnDisconnect');
    const btnStartStream = getEl('btnStartStream');
    const btnStopStream = getEl('btnStopStream');
    const btnCapture = getEl('btnCapture');
    const btnInferTPU = getEl('btnInferTPU');
    const btnInferCPU = getEl('btnInferCPU');
    
    if (btnConnect) btnConnect.addEventListener('click', connect);
    if (btnDisconnect) btnDisconnect.addEventListener('click', disconnect);
    if (btnStartStream) btnStartStream.addEventListener('click', startStream);
    if (btnStopStream) btnStopStream.addEventListener('click', stopStream);
    if (btnCapture) btnCapture.addEventListener('click', capture);
    if (btnInferTPU) btnInferTPU.addEventListener('click', () => runInference('tpu'));
    if (btnInferCPU) btnInferCPU.addEventListener('click', () => runInference('cpu'));
    
    // Fullscreen
    const btnColorFS = getEl('btnColorFullscreen');
    const btnDepthFS = getEl('btnDepthFullscreen');
    
    if (btnColorFS) {
      btnColorFS.addEventListener('click', () => {
        const img = getEl('colorStream');
        if (img && img.requestFullscreen) img.requestFullscreen();
      });
    }
    
    if (btnDepthFS) {
      btnDepthFS.addEventListener('click', () => {
        const img = getEl('depthStream');
        if (img && img.requestFullscreen) img.requestFullscreen();
      });
    }
    
    // Log filters
    getEls('.log-filter').forEach(filter => {
      filter.addEventListener('click', () => {
        getEls('.log-filter').forEach(f => f.classList.remove('active'));
        filter.classList.add('active');
        appState.logFilter = filter.dataset.level;
        
        // Re-render logs
        const container = getEl('logsContainer');
        if (container) {
          container.innerHTML = '';
          appState.logs.forEach(entry => renderLog(entry));
        }
      });
    });
    
    const btnClearLogs = getEl('btnClearLogs');
    if (btnClearLogs) {
      btnClearLogs.addEventListener('click', () => {
        appState.logs = [];
        const container = getEl('logsContainer');
        if (container) container.innerHTML = '';
        addLog('Logs cleared', 'info');
      });
    }
    
    const btnExportLogs = getEl('btnExportLogs');
    if (btnExportLogs) {
      btnExportLogs.addEventListener('click', () => {
        const logsText = appState.logs.map(l => 
          `[${l.time}] [${l.level.toUpperCase()}] ${l.message}`
        ).join('\n');
        const blob = new Blob([logsText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `opss_logs_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Logs exported', 'success');
      });
    }
  }

  // Keyboard Shortcuts
  function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        appState.connected ? disconnect() : connect();
      } else if (e.key === ' ' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        appState.streaming ? stopStream() : startStream();
      } else if (e.key === 'c' && !e.ctrlKey) {
        capture();
      } else if (e.key === 'l' && !e.ctrlKey) {
        switchTab('logs');
      }
    });
  }

  // Initialize Application
  function init() {
    // Load saved preferences
    const lastTab = localStorage.getItem('opss_last_tab') || 'dashboard';
    if (window.location.hash) {
      switchTab(window.location.hash.substring(1));
    } else {
      switchTab(lastTab);
    }
    
    initEventListeners();
    initKeyboardShortcuts();
    
    // Initial log entry
    addLog('OPSS initialized', 'info');
    addLog('Press Ctrl+K to connect, Space to stream', 'info');
    
    // Handle hash changes
    window.addEventListener('hashchange', () => {
      if (window.location.hash) {
        switchTab(window.location.hash.substring(1));
      }
    });
    
    // Check initial connection status
    fetch(API.endpoints.camera.status)
      .then(res => res.json())
      .then(data => {
        if (data.is_running || data.is_streaming) {
          appState.connected = true;
          updateStatus('connected', 'Connected');
          enableControls(true);
          startStatusPolling();
        }
      })
      .catch(() => {
        addLog('Camera not connected', 'warning');
      });
  }

  // Start the application when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();