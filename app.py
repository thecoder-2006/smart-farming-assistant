import io
import json
import os
import urllib.request

import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dotenv_file():
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv_file()
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
MODEL_DIR = os.path.join(BASE_DIR, "ai model")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.pth")
CLASS_PATH = os.path.join(BASE_DIR, "disease_classes.json")

app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["JSON_SORT_KEYS"] = False
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip() or os.getenv("HUGGING_FACE_API_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


class_names = []
num_classes = 0
model = None
model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image transformation used by the trained ResNet model
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_class_names():
    global class_names, num_classes
    if os.path.exists(CLASS_PATH):
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as file:
                class_names = json.load(file)
            num_classes = len(class_names)
            print(f"Loaded {num_classes} disease classes from {CLASS_PATH}")
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error reading disease_classes.json: {exc}")

    class_names = []
    num_classes = 0
    print("No disease class list found; model prediction will use generated class names if needed.")


load_class_names()


def load_model():
    global model, model_loaded, num_classes, class_names

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        model_loaded = False
        return False

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        checkpoint_num_classes = None

        if isinstance(checkpoint, dict) and "fc.weight" in checkpoint:
            checkpoint_num_classes = checkpoint["fc.weight"].shape[0]
            print(f"Checkpoint expects {checkpoint_num_classes} classes.")

        if checkpoint_num_classes is not None and num_classes == 0:
            num_classes = checkpoint_num_classes
            if not class_names or len(class_names) != num_classes:
                class_names = [f"class_{idx}" for idx in range(num_classes)]
        elif checkpoint_num_classes is not None and num_classes != checkpoint_num_classes:
            print(
                f"Loaded class count ({num_classes}) does not match checkpoint ({checkpoint_num_classes}). "
                "Using checkpoint class count."
            )
            num_classes = checkpoint_num_classes
            if not class_names or len(class_names) != num_classes:
                class_names = [f"class_{idx}" for idx in range(num_classes)]

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        model_loaded = True
        print("Model loaded successfully!")
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error loading model: {exc}")
        model_loaded = False
        return False


def build_fallback_remedy(disease_name, confidence, language):
    safe_disease_name = disease_name.replace("___", " - ").replace("_", " ")
    normalized = safe_disease_name.lower()

    if "healthy" in normalized:
        if language == "bn":
            return f'"{safe_disease_name}" স্বাস্থ্যকর অবস্থায় আছে ({confidence:.1f}%)। ফসলের বৃদ্ধি বজায় রাখতে নিয়মিত পানি, সুষম সার, এবং পাতার পরিচ্ছন্নতা বজায় রাখুন।'
        return f'"{safe_disease_name}" appears healthy with {confidence:.1f}% confidence. Maintain consistent irrigation, balanced nutrition, and good canopy hygiene to keep the crop productive and strong.'

    disease_map = {
        "early blight": {
            "en": 'Detected "Early Blight" with {confidence:.1f}% confidence. Remove all infected leaves immediately, increase spacing for better airflow, and apply a copper-based or Mancozeb fungicide according to label instructions to stop secondary spread.',
            "bn": '"Early Blight" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত পাতা দ্রুত তুলে ফেলুন, পাতার মধ্যে বায়ু চলাচল বাড়াতে ঘনভাবে থাকা অংশ কমিয়ে দিন, এবং কপার বা মানকোজেব ভিত্তিক ছত্রাকনাশক নির্দেশিত মাত্রায় ব্যবহার করুন যাতে রোগ দ্রুত ছড়াতে না পারে।'
        },
        "late blight": {
            "en": 'Detected "Late Blight" with {confidence:.1f}% confidence. Remove diseased foliage promptly, reduce leaf wetness and humidity around the canopy, and use a recommended systemic fungicide such as Metalaxyl or Mancozeb at the proper interval to protect remaining healthy tissue.',
            "bn": '"Late Blight" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত পাতাগুলো দ্রুত তুলে ফেলুন, গাছের চারপাশে আর্দ্রতা কম রাখুন, এবং সিস্টেমিক ছত্রাকনাশক যেমন মেটালাক্সিল বা মানকোজেব নির্দেশিত সময়সূচিতে ব্যবহার করুন যাতে বাকি গাছ সুরক্ষিত থাকে।'
        },
        "leaf mold": {
            "en": 'Detected "Leaf Mold" with {confidence:.1f}% confidence. Prune dense foliage, avoid overhead irrigation, and apply a copper or chlorothalonil-based fungicide while keeping the field well ventilated and dry during the evening hours.',
            "bn": '"Leaf Mold" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। ঘন পাতার অংশ ছাঁটাই করুন, উপরের দিক থেকে পানি দেওয়া বন্ধ করুন, এবং কপার বা ক্লোরোথালোনিল ভিত্তিক ছত্রাকনাশক ব্যবহার করুন। রাতে পাতায় ভেজা থাকে এড়িয়ে বায়ু চলাচল বজায় রাখুন।'
        },
        "septoria leaf spot": {
            "en": 'Detected "Septoria Leaf Spot" with {confidence:.1f}% confidence. Remove lower infected leaves, water at the root zone instead of overhead, and apply a broad-spectrum fungicide in a preventive schedule to reduce disease pressure in the canopy.',
            "bn": '"Septoria Leaf Spot" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। নিচের আক্রান্ত পাতা তুলে ফেলুন, উপরে থেকে নয় বরং শিকড়ের কাছে পানি দিন, এবং ব্রড-স্পেকট্রাম ছত্রাকনাশক প্রতিরোধমূলকভাবে প্রয়োগ করুন যাতে ছত্রাকের চাপ কমে।'
        },
        "powdery mildew": {
            "en": 'Detected "Powdery Mildew" with {confidence:.1f}% confidence. Improve air circulation, remove heavily infected shoots, and apply sulfur or potassium bicarbonate fungicide on a regular schedule until the white powdery growth is under control.',
            "bn": '"Powdery Mildew" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। বায়ু চলাচল বাড়ান, বেশি আক্রান্ত শাখা ছাঁটাই করুন, এবং সালফার বা পটাশিয়াম বাইকার্বোনেট ভিত্তিক ছত্রাকনাশক নিয়মিত ব্যবহার করুন যতক্ষণ না সাদা চিলতে ভাব কমে যায়।'
        },
        "bacterial spot": {
            "en": 'Detected "Bacterial Spot" with {confidence:.1f}% confidence. Remove infected foliage, avoid splashing water on leaves, disinfect pruning tools between cuts, and use a copper bactericide on a label-based schedule to slow the spread.',
            "bn": '"Bacterial Spot" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত পাতা তুলে ফেলুন, পাতায় পানি ছিটকে পড়া বন্ধ করুন, ছাঁটাইয়ের হাতিয়ার জীবাণুমুক্ত রাখুন, এবং কপার ব্যাকটেরিসাইড নির্দেশিত মাত্রায় ব্যবহার করুন যাতে রোগ দ্রুত ছড়াতে না পারে।'
        },
        "leaf spot": {
            "en": 'Detected "Leaf Spot" with {confidence:.1f}% confidence. Remove affected leaves promptly, improve spacing and ventilation, and apply a copper- or Mancozeb-based fungicide to reduce lesion expansion and protect the remaining canopy.',
            "bn": '"Leaf Spot" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত পাতা দ্রুত তুলে ফেলুন, ফসলে বায়ু চলাচল বাড়ান, এবং কপার বা মানকোজেব ভিত্তিক ছত্রাকনাশক ব্যবহার করুন যাতে ছত্রাকের লসিকাগুলো আরও ছড়াতে না পারে।'
        },
        "rust": {
            "en": 'Detected "Rust" with {confidence:.1f}% confidence. Remove older infected leaves, maintain balanced nitrogen feeding, and apply a rust-specific fungicide at the first visible pustule stage to prevent rapid field spread.',
            "bn": '"Rust" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। পুরোনো আক্রান্ত পাতা তুলে ফেলুন, নাইট্রোজেনের ভারসাম্য ঠিক রাখুন, এবং প্রথম দাগ দেখা মাত্রই রাস্ট-নির্দিষ্ট ফাঙ্গিসাইড ব্যবহার করুন যাতে দ্রুত বিস্তার না হয়।'
        },
        "cercospora": {
            "en": 'Detected "Cercospora Leaf Spot" with {confidence:.1f}% confidence. Prune infected leaves early, reduce excessive moisture in the canopy, and use a fungicide active against Cercospora to prevent worsening under humid conditions.',
            "bn": '"Cercospora Leaf Spot" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত পাতা দ্রুত ছাঁটাই করুন, পাতার ভেজা ভাব কম রাখুন, এবং সেরকোস্পোরা দমনকারী ছত্রাকনাশক ব্যবহার করুন যাতে আর্দ্র আবহাওয়ায় রোগ আরো খারাপ না হয়।'
        },
        "yellow leaf curl": {
            "en": 'Detected "Yellow Leaf Curl" with {confidence:.1f}% confidence. Remove severely infected plants, control whitefly populations, and maintain weed-free, vigorous crop conditions to reduce virus pressure and secondary spread.',
            "bn": '"Yellow Leaf Curl" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। বেশি আক্রান্ত গাছ তুলে ফেলুন, সাদা মাছি দমন করুন, এবং আগাছামুক্ত শক্তিশালী ফসলের পরিবেশ বজায় রাখুন যাতে ভাইরাসের ছড়াছড়ি কমে।'
        },
        "mosaic": {
            "en": 'Detected "Mosaic Virus" with {confidence:.1f}% confidence. Remove infected plants promptly, sanitize tools between cuts, and avoid moving from diseased tissue to healthy plants so the virus does not spread to the remaining stand.',
            "bn": '"Mosaic Virus" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত গাছ দ্রুত তুলে ফেলুন, ছাঁটাই করার আগে হাতিয়ার জীবাণুমুক্ত রাখুন, এবং সুস্থ গাছের দিকে কাজ করার আগে আক্রান্ত অংশের সংস্পর্শ এড়িয়ে চলুন যাতে ভাইরাস ছড়াতে না পারে।'
        }
    }

    for key, advice in disease_map.items():
        if key in normalized:
            return advice["bn" if language == "bn" else "en"].format(confidence=confidence)

    if language == "bn":
        return f'"{safe_disease_name}" রোগ শনাক্ত হয়েছে ({confidence:.1f}%)। আক্রান্ত পাতা তুলে ফেলুন, বায়ু চলাচল বাড়ান, এবং কপার-ভিত্তিক বা ব্রড-স্পেকট্রাম ছত্রাকনাশক ব্যবহার করুন যাতে রোগের বিস্তার কম হয়।'
    return f'Detected "{safe_disease_name}" with {confidence:.1f}% confidence. Remove infected foliage, improve airflow, and apply an appropriate copper- or broad-spectrum fungicide to limit spread and protect the remaining canopy.'


@app.route("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": model_loaded,
        "num_classes": num_classes,
        "device": str(device),
    })


@app.route("/api/config")
def app_config():
    return jsonify({
        "hasHfKey": bool(HF_API_TOKEN),
        "hasGeminiKey": bool(GEMINI_API_KEY),
        "modelReady": model_loaded,
        "provider": "huggingface" if HF_API_TOKEN else "fallback",
    })


@app.route("/api/remedy", methods=["POST"])
def remedy():
    payload = request.get_json(silent=True) or {}
    disease_name = str(payload.get("diseaseName") or "unknown crop condition")
    confidence = float(payload.get("confidence") or 0)
    language = str(payload.get("language") or "en")
    safe_disease_name = disease_name.replace("___", " - ").replace("_", " ")

    if confidence < 30 or "not a plant" in safe_disease_name.lower() or "not a leaf" in safe_disease_name.lower():
        if language == "bn":
            fallback = "এই ছবিটিতে গাছ বা পাতার বৈশিষ্ট্য শনাক্ত হয়নি। দয়া করে পরিষ্কার, কাছ থেকে তোলা সবুজ পাতার ছবি আপলোড করুন।"
        else:
            fallback = "This image does not appear to be a plant leaf or crop sample. Please upload a clear, close-up photo of a plant leaf or crop leaf."
        return jsonify({"success": True, "remedy": fallback, "source": "validation"})

    if not HF_API_TOKEN:
        return jsonify({"success": True, "remedy": build_fallback_remedy(safe_disease_name, confidence, language), "source": "fallback"})

    prompt = (
        f"You are an expert agricultural scientist helping farmers. "
        f"Disease: {safe_disease_name}. Confidence: {confidence:.1f}%. "
        "Give practical, safe, and concise remedy advice in 2 short sentences. "
        "Mention crop sanitation, airflow, and a suitable treatment if needed."
    )

    hf_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {"max_new_tokens": 96, "temperature": 0.7, "do_sample": True},
        "options": {"wait_for_model": True}
    }).encode("utf-8")
    request_obj = urllib.request.Request(
        hf_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        text = ""
        if isinstance(response_data, list):
            first = response_data[0] if response_data else {}
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("summary_text") or str(first)
        elif isinstance(response_data, dict):
            text = response_data.get("generated_text") or response_data.get("summary_text") or response_data.get("error") or json.dumps(response_data)

        if not text:
            raise ValueError("Hugging Face returned an empty result")

        if "\n" in text:
            text = text.splitlines()[0].strip()

        return jsonify({"success": True, "remedy": text.strip(), "source": "huggingface"})
    except Exception as exc:
        return jsonify({"success": True, "remedy": build_fallback_remedy(safe_disease_name, confidence, language), "source": "fallback", "error": str(exc)})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Training may still be in progress."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        image_base64 = __import__("base64").b64encode(image_buffer.getvalue()).decode()

        tensor = image_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item() * 100

        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"class_{predicted_class_idx}"
        top_k = min(5, len(class_names))
        top5_probs, top5_indices = torch.topk(probabilities[0], k=top_k)
        top5_predictions = [
            {
                "class": class_names[idx.item()] if idx.item() < len(class_names) else f"class_{idx.item()}",
                "confidence": prob.item() * 100,
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        if confidence < 30:
            return jsonify({
                "success": True,
                "prediction": "Not a plant or leaf",
                "confidence": confidence,
                "isNotPlantLeaf": True,
                "top5": top5_predictions,
                "image": f"data:image/png;base64,{image_base64}",
            })

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "isNotPlantLeaf": False,
            "top5": top5_predictions,
            "image": f"data:image/png;base64,{image_base64}",
        })
    except Exception as exc:
        return jsonify({"error": f"Error processing image: {str(exc)}"}), 500


@app.route("/<path:path>")
def serve_static(path):
    if path.startswith("api/"):
        return jsonify({"error": "API endpoint not found"}), 404
    return send_from_directory(PUBLIC_DIR, path)


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Flask app on http://{host}:{port}")
    app.run(debug=False, host=host, port=port)
