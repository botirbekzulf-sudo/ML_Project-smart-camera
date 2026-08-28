/**
 * api.js
 * Логика обмена с Flask backend: классификация фото, поиск в Google,
 * предсказание ценового класса. Все три шага сценария завязаны здесь.
 */

const API_BASE = ""; // сайт и API на одном хосте (Flask раздаёт и фронтенд)

const resultPanel = document.getElementById("resultPanel");
const geminiVisionText = document.getElementById("geminiVisionText");
const modelForm = document.getElementById("modelForm");
const modelInput = document.getElementById("modelInput");
const googleLink = document.getElementById("googleLink");
const priceCard = document.getElementById("priceCard");
const priceClassValue = document.getElementById("priceClassValue");
const probaBars = document.getElementById("probaBars");
const featuresUsed = document.getElementById("featuresUsed");
const specsCard = document.getElementById("specsCard");
const specsBars = document.getElementById("specsBars");
const lookupError = document.getElementById("lookupError");
const manualRam = document.getElementById("manualRam");
const manualBattery = document.getElementById("manualBattery");
const manualStorage = document.getElementById("manualStorage");
const manualCamera = document.getElementById("manualCamera");

/**
 * Шаг 1: кадр с камеры -> Gemini 3.6 (мультимодально) -> текстовый ответ
 * выводится ПРЯМО на сайте. Вызывается из camera.js на каждый кадр в
 * реальном времени (см. GEMINI_SCAN_INTERVAL_MS).
 */
window.handleCapturedImage = async function handleCapturedImage(imageBlob) {
  resultPanel.hidden = false;

  const formData = new FormData();
  formData.append("image", imageBlob, "scan.jpg");

  try {
    const response = await fetch(`${API_BASE}/api/gemini-vision`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.error) {
      geminiVisionText.textContent = `Gemini недоступен: ${data.error}`;
      return;
    }

    geminiVisionText.textContent = data.answer;
  } catch (err) {
    geminiVisionText.textContent = `Ошибка соединения с сервером: ${err}`;
  }
};

/** Шаг 2: название модели -> характеристики + ссылка на Google */
/** Шаг 2: название модели -> Gemini проверяет, что это устройство ->
 *  характеристики + ссылка на Google. */
modelForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const modelName = modelInput.value.trim();
  if (!modelName) return;

  lookupError.hidden = true;
  googleLink.hidden = true;
  priceCard.hidden = true;
  specsCard.hidden = true;

  try {
    const response = await fetch(`${API_BASE}/api/lookup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      // Gemini решил, что это не техника (или другая ошибка запроса) —
      // дальше по пайплайну (Google + ML) не идём.
      lookupError.textContent = data.error || "Не удалось выполнить поиск.";
      lookupError.hidden = false;
      return;
    }

    if (data.source_url) {
      googleLink.href = data.source_url;
      googleLink.hidden = false;
      // Как описано в задании: сайт открывает Google с характеристиками устройства.
      window.open(data.source_url, "_blank", "noopener");
    }

    // Шаг 3: если пользователь вписал характеристики вручную — используем
    // их (даёт точный и РАЗНЫЙ результат для разных устройств). Иначе —
    // то, что нашлось в raw_text (без платного Google API оно обычно
    // пустое, и тогда модель работает на одних значениях по умолчанию —
    // отсюда одинаковый класс для всех устройств).
    const manualForm = collectManualSpecs();
    if (manualForm) {
      await predictPriceFromForm(manualForm);
    } else {
      await predictPriceFromText(data.raw_text || "");
    }
  } catch (err) {
    lookupError.textContent = `Ошибка соединения: ${err}`;
    lookupError.hidden = false;
  }
});

/** Собирает заполненные вручную поля характеристик. null, если все пустые. */
function collectManualSpecs() {
  const ramGb = parseFloat(manualRam.value);
  const batteryMah = parseFloat(manualBattery.value);
  const storageGb = parseFloat(manualStorage.value);
  const cameraMp = parseFloat(manualCamera.value);

  if (![ramGb, batteryMah, storageGb, cameraMp].some((v) => !isNaN(v) && v > 0)) {
    return null;
  }

  const form = {};
  if (!isNaN(ramGb)) form.ram = ramGb * 1024; // модель ждёт RAM в МБ
  if (!isNaN(batteryMah)) form.battery_power = batteryMah;
  if (!isNaN(storageGb)) form.int_memory = storageGb;
  if (!isNaN(cameraMp)) form.pc = cameraMp;
  return form;
}

async function predictPriceFromForm(form) {
  try {
    const response = await fetch(`${API_BASE}/api/predict-price`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ form }),
    });
    const data = await response.json();
    if (data.error) return;
    renderPriceResult(data);
  } catch (err) {
    console.error(err);
  }
}

async function predictPriceFromText(rawText) {
  try {
    const response = await fetch(`${API_BASE}/api/predict-price`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawText }),
    });
    const data = await response.json();
    if (data.error) return;
    renderPriceResult(data);
  } catch (err) {
    console.error(err);
  }
}

function renderPriceResult(data) {
  priceCard.hidden = false;
  priceClassValue.textContent = data.price_class_name;

  // Тот же визуальный стиль шкал, что и у карточки "Характеристики устройства":
  // подпись + процент + полоса заполненности. Предсказанный класс подсвечен.
  probaBars.innerHTML = "";
  Object.entries(data.probabilities).forEach(([label, value]) => {
    const percent = Math.round(value * 100);
    const isPredicted = label === data.price_class_name;
    const row = document.createElement("div");
    row.className = "specs-row" + (isPredicted ? " predicted" : "");
    row.innerHTML = `
      <div class="specs-row-head">
        <span class="specs-row-label">${label}${isPredicted ? " ✓" : ""}</span>
        <span class="specs-row-value">${percent}%</span>
      </div>
      <div class="specs-track">
        <div class="specs-fill" style="width:${percent}%"></div>
      </div>
    `;
    probaBars.appendChild(row);
  });

  featuresUsed.textContent = JSON.stringify(data.features_used, null, 2);

  if (Array.isArray(data.features_display)) {
    renderSpecsBars(data.features_display);
  }
}

/** Красивые шкалы заполненности для характеристик устройства. */
function renderSpecsBars(rows) {
  specsCard.hidden = false;
  specsBars.innerHTML = "";

  rows.forEach((row) => {
    const el = document.createElement("div");
    el.className = "specs-row";
    const unit = row.unit ? ` ${row.unit}` : "";
    el.innerHTML = `
      <div class="specs-row-head">
        <span class="specs-row-label">${row.label}</span>
        <span class="specs-row-value">${row.value}${unit}</span>
      </div>
      <div class="specs-track">
        <div class="specs-fill" style="width:${row.percent}%"></div>
      </div>
      <div class="specs-row-percent">${row.percent}%</div>
    `;
    specsBars.appendChild(el);
  });
}
