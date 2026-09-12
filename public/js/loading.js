/* ==========================================================================
   Loading Video & Modal Animation Controller
   ========================================================================== */

const loadingModal = document.getElementById('loading-modal');
const loadingVideo = document.getElementById('loading-video');
const loadingText = document.getElementById('loading-text');

const scanningMessages = {
    en: [
        "Initializing camera capture...",
        "Scanning leaf surface texture...",
        "Preprocessing image tensor (224x224)...",
        "Running MobileNetV2 Neural Network inference...",
        "Querying Gemini API for localized remedy...",
        "Generating diagnostic report and recommendations..."
    ],
    bn: [
        "ক্যামেরা ক্যাপচার প্রস্তুত করা হচ্ছে...",
        "পাতার উপরিভাগের গঠন স্ক্যান করা হচ্ছে...",
        "ইমেজ টেনসর প্রসেস করা হচ্ছে (২২৪x২২৪)...",
        "মোবাইলনেট-ভি২ নিউরাল নেটওয়ার্ক চালনা করা হচ্ছে...",
        "জেমিয়াই এআই থেকে উপযুক্ত প্রতিকার খোঁজা হচ্ছে...",
        "রোগ নির্ণয় রিপোর্ট এবং প্রতিকার তৈরি করা হচ্ছে..."
    ]
};

let messageInterval = null;

function showLoadingModal(lang = 'en') {
    if (!loadingModal) return;
    
    loadingModal.classList.remove('hidden');
    if (loadingVideo) {
        loadingVideo.currentTime = 0;
        loadingVideo.play().catch(() => {});
    }

    let index = 0;
    const messages = scanningMessages[lang] || scanningMessages['en'];
    if (loadingText) loadingText.innerText = messages[0];

    messageInterval = setInterval(() => {
        index = (index + 1) % messages.length;
        if (loadingText) loadingText.innerText = messages[index];
    }, 800);
}

function hideLoadingModal() {
    if (!loadingModal) return;
    
    loadingModal.classList.add('hidden');
    if (loadingVideo) {
        loadingVideo.pause();
    }
    if (messageInterval) {
        clearInterval(messageInterval);
        messageInterval = null;
    }
}