/**
 * camera.js
 * Доступ к камере устройства + непрерывное сканирование в реальном времени:
 * пока камера включена, кадр автоматически уходит в Gemini 3.6 (backend
 * /api/gemini-vision) каждые SCAN_INTERVAL_MS миллисекунд, и его текстовый
 * ответ выводится прямо на сайте. Интервал больше, чем был бы для лёгкой
 * локальной модели — мультимодальный запрос к Gemini дороже и медленнее.
 * Кнопка "Сканировать" включает/выключает цикл вручную. Также поддерживается
 * загрузка файла как запасной вариант, если камеры нет.
 */

const SCAN_INTERVAL_MS = 3000;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const fileInput = document.getElementById("fileInput");
const statusBadge = document.getElementById("statusBadge");

let currentStream = null;
let scanTimer = null;
let requestInFlight = false; // защита от наложения запросов, если сеть медленная

async function startCamera() {
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    video.srcObject = currentStream;
    statusBadge.textContent = "Камера готова";
    captureBtn.disabled = false;
  } catch (err) {
    console.warn("Камера недоступна:", err);
    statusBadge.textContent = "Камера недоступна — загрузите фото";
  }
}

function captureFrameAsBlob() {
  return new Promise((resolve) => {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
  });
}

async function scanTick() {
  if (requestInFlight) return; // пропускаем тик, если предыдущий кадр ещё обрабатывается
  requestInFlight = true;
  try {
    const blob = await captureFrameAsBlob();
    await window.handleCapturedImage(blob);
  } finally {
    requestInFlight = false;
  }
}

function startLiveScanning() {
  if (!currentStream || scanTimer) return;
  scanTimer = setInterval(scanTick, SCAN_INTERVAL_MS);
  statusBadge.textContent = "Сканирование в реальном времени";
  statusBadge.classList.add("live");
  captureBtn.textContent = "⏸ Остановить сканирование";
  scanTick(); // не ждать первый интервал
}

function stopLiveScanning() {
  clearInterval(scanTimer);
  scanTimer = null;
  statusBadge.textContent = "Камера готова";
  statusBadge.classList.remove("live");
  captureBtn.textContent = "▶ Сканировать";
}

captureBtn.addEventListener("click", () => {
  if (scanTimer) stopLiveScanning();
  else startLiveScanning();
});

// Пауза сканирования, если вкладка свёрнута — экономим запросы и батарею.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopLiveScanning();
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  stopLiveScanning();
  await window.handleCapturedImage(file);
});

captureBtn.disabled = true;
captureBtn.textContent = "▶ Сканировать";
startCamera();
