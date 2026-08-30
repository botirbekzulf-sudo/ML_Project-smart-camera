/**
 * camera.js
 * Доступ к камере устройства + сканирование в реальном времени: пока
 * камера включена, кадр автоматически уходит в Gemini 3.6 (backend
 * /api/gemini-vision) каждые SCAN_INTERVAL_MS миллисекунд.
 *
 * ЭКОНОМИЯ КВОТЫ GEMINI (бесплатный тариф ограничен и по запросам в
 * минуту, и по запросам в сутки — сканер камеры расходует её быстрее
 * всего, так как работает непрерывно):
 *   - интервал между кадрами специально не маленький (10 сек);
 *   - сканер САМ останавливается после MAX_AUTO_CAPTURES кадров, чтобы
 *     не расходовать квоту в фоне, если забыли нажать "Остановить";
 *   - при ошибке 429 (лимит исчерпан) сканер делает более длинную
 *     паузу (COOLDOWN_MS) вместо повторных попыток в том же темпе.
 */

const SCAN_INTERVAL_MS = 10000;  // интервал между кадрами
const COOLDOWN_MS = 30000;       // пауза после 429, пока не "остынет" лимит
const MAX_AUTO_CAPTURES = 6;     // автостоп после N кадров подряд (~1 минута)

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const fileInput = document.getElementById("fileInput");
const statusBadge = document.getElementById("statusBadge");

let currentStream = null;
let scanTimer = null;
let requestInFlight = false; // защита от наложения запросов, если сеть медленная
let cooldownUntil = 0;       // timestamp, до которого сканер "отдыхает" после 429
let autoCaptureCount = 0;    // счётчик кадров с начала текущей сессии сканирования

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
  if (Date.now() < cooldownUntil) return; // ещё "остываем" после лимита

  if (autoCaptureCount >= MAX_AUTO_CAPTURES) {
    stopLiveScanning();
    statusBadge.textContent = `Автосканирование остановлено (экономия квоты) — нажми "Сканировать"`;
    return;
  }

  requestInFlight = true;
  autoCaptureCount += 1;
  try {
    const blob = await captureFrameAsBlob();
    const hitRateLimit = await window.handleCapturedImage(blob);
    if (hitRateLimit) {
      cooldownUntil = Date.now() + COOLDOWN_MS;
      statusBadge.textContent = `Пауза ${COOLDOWN_MS / 1000}с (лимит Gemini)`;
    } else if (scanTimer) {
      statusBadge.textContent = "Сканирование в реальном времени";
    }
  } finally {
    requestInFlight = false;
  }
}

function startLiveScanning() {
  if (!currentStream || scanTimer) return;
  autoCaptureCount = 0;
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
