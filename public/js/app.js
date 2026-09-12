/* ==========================================================================
   Main Application Logic, Gemini API Integration, Telemetry & Speech
   ========================================================================== */

// ---------------------------------------------------------------------------
// Gemini API — loaded dynamically (ES module import via dynamic import)
// ---------------------------------------------------------------------------
let ai = null;

async function initAiConfig() {
    try {
        const configResponse = await fetch('/api/config');
        const config = await configResponse.json();
        if (config && config.provider === 'huggingface') {
            console.log('Hugging Face remedy backend is available.');
        } else {
            console.warn('No AI provider configured. Remedy service will use safe fallback guidance.');
        }
    } catch (e) {
        console.warn('AI provider check failed; using local fallback guidance.', e);
    }
}

initAiConfig();

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentLang = 'en';
let activePumpState = false;

// ---------------------------------------------------------------------------
// UI Element References
// ---------------------------------------------------------------------------
const imageInput = document.getElementById('image-input');
const openCameraBtn = document.getElementById('btn-open-camera');
const chooseImageBtn = document.getElementById('btn-choose-image');
const imagePreview = document.getElementById('image-preview');
const btnAnalyze = document.getElementById('btn-analyze');

const resHealthStatus = document.getElementById('res-health-status');
const resDiseaseName = document.getElementById('res-disease-name');
const resPestStatus = document.getElementById('res-pest-status');
const resConfidence = document.getElementById('res-confidence');
const resRecommendation = document.getElementById('res-recommendation');

// ---------------------------------------------------------------------------
// Image Upload Handler
// ---------------------------------------------------------------------------
if (imageInput) {
    imageInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (event) {
                imagePreview.src = event.target.result;
                imagePreview.style.display = 'block';
                btnAnalyze.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });
}

function triggerFilePicker(useCamera) {
    if (!imageInput) return;

    imageInput.value = '';
    if (useCamera) {
        imageInput.setAttribute('capture', 'environment');
        imageInput.setAttribute('accept', 'image/*');
    } else {
        imageInput.removeAttribute('capture');
        imageInput.setAttribute('accept', 'image/*');
    }

    setTimeout(function () {
        imageInput.click();
    }, 50);
}

if (openCameraBtn && imageInput) {
    openCameraBtn.addEventListener('click', function (e) {
        e.preventDefault();
        triggerFilePicker(true);
    });
}

if (chooseImageBtn && imageInput) {
    chooseImageBtn.addEventListener('click', function (e) {
        e.preventDefault();
        triggerFilePicker(false);
    });
}

// ---------------------------------------------------------------------------
// Gemini Remedy Fetcher
// ---------------------------------------------------------------------------

/**
 * Fetches real-time dynamic remedies from Gemini API based on detected disease
 * @param {string} diseaseName - Detected plant label
 * @param {string} language - Current UI language ('en' or 'bn')
 * @returns {Promise<string>} - Generated recommendation text
 */
function getNotPlantFeedback(language) {
    return language === 'bn'
        ? "এই ছবিটিতে গাছ বা পাতার বৈশিষ্ট্য শনাক্ত হয়নি। দয়া করে পরিষ্কার, কাছ থেকে তোলা সবুজ পাতার ছবি আপলোড করুন।"
        : "This image does not appear to be a plant leaf or crop sample. Please upload a clear, close-up photo of a plant leaf or crop leaf.";
}

async function fetchGeminiRemedy(diseaseName, confidence, language) {
    const safeDiseaseName = String(diseaseName || 'unknown crop condition').replace(/___/g, ' - ').replace(/_/g, ' ');
    const confidenceValue = Number(confidence) || 0;

    if (confidenceValue < 30 || safeDiseaseName.toLowerCase().includes('not a plant') || safeDiseaseName.toLowerCase().includes('not a leaf')) {
        return getNotPlantFeedback(language);
    }

    try {
        const response = await fetch('/api/remedy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                diseaseName: safeDiseaseName,
                confidence: confidenceValue,
                language,
            }),
        });

        const data = await response.json();
        if (data && data.remedy) {
            return data.remedy;
        }
    } catch (error) {
        console.error('Backend remedy request failed:', error);
    }

    return language === 'bn'
        ? `"${safeDiseaseName}" রোগ শনাক্ত হয়েছে (${confidenceValue.toFixed(1)}%)। আক্রান্ত পাতা তুলে ফেলুন, বায়ুচলাচল ঠিক রাখুন এবং কপার অক্সিক্লোরাইড ৫০% ডাব্লুপি (৪ গ্রাম/লিটার জলে) স্প্রে করুন।`
        : `Detected "${safeDiseaseName}" with ${confidenceValue.toFixed(1)}% confidence. Remove infected leaves, improve airflow, and apply Copper Oxychloride 50% WP at 4g/L of water.`;
}

// ---------------------------------------------------------------------------
// Analyze Button
// ---------------------------------------------------------------------------
if (btnAnalyze) {
    btnAnalyze.addEventListener('click', async function () {
        showLoadingModal(currentLang);

        try {
            // 1. Run server-side prediction via Flask API
            const result = await runCropInference(imagePreview);

            hideLoadingModal();

            if (result.isNotPlantLeaf) {
                resHealthStatus.innerText = currentLang === 'bn' ? "গাছ/পাতা শনাক্ত হয়নি" : "No plant or leaf detected";
                resDiseaseName.innerText = currentLang === 'bn' ? "অনুপযুক্ত ছবি" : "Invalid image";
                resPestStatus.innerText = currentLang === 'bn' ? "কোনো রোগ বা কীট শনাক্ত হয়নি" : "No disease or pest detected";
                resConfidence.innerText = `${result.confidence}%`;
                resRecommendation.innerText = getNotPlantFeedback(currentLang);
                document.getElementById('disease-result-area').scrollIntoView({ behavior: 'smooth' });
                return;
            }

            // 2. Fetch dynamic remedies using Gemini 2.5 Flash and the disease confidence
            const dynamicRemedy = await fetchGeminiRemedy(result.diseaseRaw, result.confidence, currentLang);

            // 3. Update UI Elements
            const isHealthy = result.diseaseRaw.toLowerCase().includes('healthy');

            resHealthStatus.innerText = isHealthy
                ? (currentLang === 'bn' ? "সুস্থ ফসল" : "Healthy Crop")
                : (currentLang === 'bn' ? "আক্রান্ত / অসুস্থ ফসল" : "Infected / Unhealthy");

            resDiseaseName.innerText = result.diseaseRaw.replace(/___/g, ' - ').replace(/_/g, ' ');
            resPestStatus.innerText = isHealthy
                ? (currentLang === 'bn' ? "কোনো পোকার আক্রমণ নেই" : "No Pest Damage Detected")
                : (currentLang === 'bn' ? "ক্ষতিকারক পোকা বা ছত্রাকের উপস্থিতি চিহ্নিত" : "Pest or Fungal Spore Activity Identified");

            resConfidence.innerText = `${result.confidence}%`;
            resRecommendation.innerText = dynamicRemedy;

            // Scroll down smoothly to show analysis outputs
            document.getElementById('disease-result-area').scrollIntoView({ behavior: 'smooth' });
        } catch (err) {
            console.error("Analysis process failed:", err);
            hideLoadingModal();
        }
    });
}

// ---------------------------------------------------------------------------
// Telemetry Sliders
// ---------------------------------------------------------------------------
window.updateTelemetry = function () {
    const moisture = document.getElementById('moisture-slider').value;
    const temp = document.getElementById('temp-slider').value;
    const humidity = document.getElementById('humidity-slider').value;

    document.getElementById('soil-moisture-val').innerText = `${moisture}%`;
    document.getElementById('temp-val').innerText = `${temp}°C`;
    document.getElementById('humidity-val').innerText = `${humidity}%`;

    // Automatic Irrigation Decision Logic
    const decisionText = document.getElementById('irrigation-decision-text');
    if (moisture < 30) {
        decisionText.innerText = currentLang === 'bn'
            ? "মাটির আর্দ্রতা কম (৩০% এর নিচে)। সেচ চালু করার পরামর্শ দেওয়া হচ্ছে।"
            : "Soil moisture is critically low (<30%). Irrigation recommended.";
        decisionText.style.color = "red";
    } else {
        decisionText.innerText = currentLang === 'bn'
            ? "মাটির আর্দ্রতা স্বাভাবিক রয়েছে। কোনো সেচের প্রয়োজন নেই।"
            : "Soil moisture is optimal. No irrigation required.";
        decisionText.style.color = "green";
    }

    // Risk Alert Logic
    const alertDrought = document.getElementById('alert-drought');
    const alertHeatwave = document.getElementById('alert-heatwave');

    if (moisture < 25 && temp > 35) {
        alertDrought.innerText = currentLang === 'bn' ? "উচ্চ খরার ঝুঁকি!" : "High Drought Risk Alert!";
        alertDrought.style.color = "red";
    } else {
        alertDrought.innerText = currentLang === 'bn' ? "স্বাভাবিক" : "Low Risk";
        alertDrought.style.color = "green";
    }

    if (temp > 40) {
        alertHeatwave.innerText = currentLang === 'bn' ? "তীব্র দাবদাহ সতর্কতা!" : "Severe Heat-Wave Alert!";
        alertHeatwave.style.color = "red";
    } else {
        alertHeatwave.innerText = currentLang === 'bn' ? "স্বাভাবিক তাপমাত্রা" : "Normal Temperature";
        alertHeatwave.style.color = "green";
    }
};

// ---------------------------------------------------------------------------
// Manual Pump Control Toggle
// ---------------------------------------------------------------------------
window.togglePump = function () {
    activePumpState = !activePumpState;
    const pumpStatusText = document.getElementById('pump-status-text');
    pumpStatusText.innerText = activePumpState ? "ON (জল সেচ চলছে)" : "OFF (বন্ধ)";
    pumpStatusText.style.color = activePumpState ? "green" : "black";
};

// ---------------------------------------------------------------------------
// Language Switcher
// ---------------------------------------------------------------------------
const translations = {
    en: {
        appTitle: 'Smart Farming Assistant',
        heroTitle: 'Smart Agriculture & Crop Health Dashboard',
        heroSub: 'AI-powered disease detection, real-time irrigation monitoring, and risk alerts for farmers.',
        cropHealthSection: '🌱 1. Crop Health & Disease Detection',
        imageCaptureTitle: '📷 Leaf / Crop Image Capture',
        uploadLabel: 'Tap to Open Camera or Choose Photo:',
        openCameraBtn: '📷 Open Camera',
        chooseImageBtn: '🖼️ Click Image',
        imagePreviewLabel: 'Image Preview:',
        analyzeBtn: '🔍 Analyze Crop Health',
        diagnosticResults: 'Diagnostic Results',
        plantHealthStatusLabel: '🌿 Plant Health Status:',
        diseaseDetectionLabel: '🦠 Crop Disease Detection:',
        pestStatusLabel: '🐛 Pest / Pest Damage Identification:',
        confidenceLabel: '🎯 Confidence Score:',
        recommendationLabel: '💡 Recommendations & Next Action:',
        recommendationDefault: 'Upload or capture a leaf photo to view treatment guidance.',
        audioGuideBtn: '🔊 Listen Audio Guidance',
        irrigationSection: '💧 2. Smart Irrigation & Soil Monitoring',
        telemetryTitle: '📊 Real-Time Field Telemetry',
        soilMoistureLabel: '🌱 Soil Moisture Monitoring',
        tempLabel: '🌡️ Temperature Monitoring',
        humidityLabel: '💧 Humidity Monitoring',
        pumpTitle: '💦 Irrigation Requirement Detection & Pump Control',
        statusLabel: 'Status:',
        pumpStateLabel: '🚰 Automatic / Remote Pump State:',
        pumpToggleBtn: 'Toggle Pump ON/OFF Manually',
        riskSection: '🌦️ 3. Agricultural Risk & Smart Alerts',
        riskTitle: '📱 Dashboard Environmental Risk Indications',
        droughtRiskLabel: '☀️ Drought Risk Alert:',
        floodRiskLabel: '🌊 Flood / Waterlogging Risk Alert:',
        heatRiskLabel: '🔥 Heat-Stress / Heat-Wave Alert:',
        weatherAssessmentLabel: '🌧️ Weather / Environment Assessment:',
        notificationsLabel: '🔔 Active Farmer Notifications:'
    },
    bn: {
        appTitle: 'স্মার্ট ফার্মিং অ্যাসিস্ট্যান্ট',
        heroTitle: 'স্মার্ট কৃষি ও ফসলের স্বাস্থ্য ড্যাশবোর্ড',
        heroSub: 'কৃষকদের জন্য এআই-চালিত রোগ শনাক্তকরণ, রিয়েল-টাইম সেচ পর্যবেক্ষণ ও ঝুঁকি সতর্কতা।',
        cropHealthSection: '🌱 ১. ফসলের স্বাস্থ্য ও রোগ শনাক্তকরণ',
        imageCaptureTitle: '📷 পাতা / ফসলের ছবি তুলুন',
        uploadLabel: 'ক্যামেরা খুলতে বা ফটো বাছাই করতে ক্লিক করুন:',
        openCameraBtn: '📷 ক্যামেরা খুলুন',
        chooseImageBtn: '🖼️ ছবি বাছাই করুন',
        imagePreviewLabel: 'ছবির পূর্বরূপ:',
        analyzeBtn: '🔍 ফসলের স্বাস্থ্য বিশ্লেষণ করুন',
        diagnosticResults: 'রোগ নির্ণয়ের ফলাফল',
        plantHealthStatusLabel: '🌿 গাছের স্বাস্থ্য অবস্থা:',
        diseaseDetectionLabel: '🦠 ফসলের রোগ শনাক্তকরণ:',
        pestStatusLabel: '🐛 কীট / কীটের ক্ষতি শনাক্তকরণ:',
        confidenceLabel: '🎯 আত্মবিশ্বাসের স্কোর:',
        recommendationLabel: '💡 সুপারিশ ও পরবর্তী পদক্ষেপ:',
        recommendationDefault: 'চিকিৎসা নির্দেশিকা দেখতে পাতা বা ফসলের ছবি আপলোড বা তুলুন।',
        audioGuideBtn: '🔊 বাংলা অডিও শুনুন',
        irrigationSection: '💧 ২. স্মার্ট সেচ ও মাটি পর্যবেক্ষণ',
        telemetryTitle: '📊 রিয়েল-টাইম মাঠ টেলিমেট্রি',
        soilMoistureLabel: '🌱 মাটির আর্দ্রতা পর্যবেক্ষণ',
        tempLabel: '🌡️ তাপমাত্রা পর্যবেক্ষণ',
        humidityLabel: '💧 আপেক্ষিক আর্দ্রতা পর্যবেক্ষণ',
        pumpTitle: '💦 সেচের প্রয়োজন ও পাম্প নিয়ন্ত্রণ',
        statusLabel: 'অবস্থা:',
        pumpStateLabel: '🚰 অটোমেটিক / রিমোট পাম্প অবস্থা:',
        pumpToggleBtn: 'পাম্প অন/অফ ম্যানুয়ালি পরিবর্তন করুন',
        riskSection: '🌦️ ৩. কৃষি ঝুঁকি ও স্মার্ট সতর্কতা',
        riskTitle: '📱 ড্যাশবোর্ড পরিবেশগত ঝুঁকি নির্দেশক',
        droughtRiskLabel: '☀️ খরা ঝুঁকি:',
        floodRiskLabel: '🌊 বন্যা / জলাবদ্ধতা ঝুঁকি:',
        heatRiskLabel: '🔥 তাপমাত্রা চাপ / তাপপ্রবাহ সতর্কতা:',
        weatherAssessmentLabel: '🌧️ আবহাওয়া / পরিবেশ মূল্যায়ন:',
        notificationsLabel: '🔔 সক্রিয় কৃষক নোটিফিকেশন:'
    }
};

function applyLanguageTranslations(lang) {
    const bundle = translations[lang] || translations.en;
    document.querySelectorAll('[data-i18n]').forEach((element) => {
        const key = element.dataset.i18n;
        if (bundle[key] !== undefined) {
            element.textContent = bundle[key];
        }
    });
}

window.setLanguage = function (lang) {
    currentLang = lang;
    document.getElementById('btn-lang-en').style.fontWeight = lang === 'en' ? 'bold' : 'normal';
    document.getElementById('btn-lang-bn').style.fontWeight = lang === 'bn' ? 'bold' : 'normal';
    applyLanguageTranslations(lang);
    window.updateTelemetry();
};

// ---------------------------------------------------------------------------
// Web Speech API Text-to-Speech Output
// ---------------------------------------------------------------------------
function getBestVoiceForLanguage(lang) {
    const synth = window.speechSynthesis;
    if (!synth) return null;

    const voices = synth.getVoices();
    if (!voices || voices.length === 0) return null;

    if (lang === 'bn') {
        return voices.find((voice) => {
            const value = (voice.lang || '').toLowerCase();
            return value.startsWith('bn') || value.includes('bengali') || value.includes('bangla');
        }) || voices.find((voice) => (voice.lang || '').toLowerCase().startsWith('en')) || voices[0];
    }

    return voices.find((voice) => (voice.lang || '').toLowerCase().startsWith('en')) || voices[0];
}

window.speakResult = function () {
    const textToSpeak = resRecommendation.innerText || document.getElementById('res-recommendation')?.innerText || '';
    if (!textToSpeak.trim() || !('speechSynthesis' in window)) return;

    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    if (!voices || voices.length === 0) {
        synth.onvoiceschanged = () => {
            synth.onvoiceschanged = null;
            window.speakResult();
        };
        return;
    }

    synth.cancel();

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    const preferredLocale = currentLang === 'bn' ? 'bn-IN' : 'en-US';
    const matchingVoice = getBestVoiceForLanguage(currentLang);

    utterance.lang = preferredLocale;
    utterance.rate = 0.85;
    utterance.pitch = currentLang === 'bn' ? 1.15 : 1.0;
    utterance.volume = 1.0;

    if (matchingVoice) {
        utterance.voice = matchingVoice;
    }

    synth.speak(utterance);
};