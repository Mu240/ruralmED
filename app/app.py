# DAYS 13-14 - app/app.py
# Run: python app/app.py
# Then open browser: http://localhost:5000

from flask import Flask, request, jsonify, render_template
import sys, os, json, datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Initialize engine once at startup
engine = None

def get_engine():
    global engine
    if engine is None:
        from rag.inference import RuralMEDEngine
        engine = RuralMEDEngine()
    return engine

# ── Routes ────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/assess", methods=["POST"])
def assess():
    """Main endpoint — takes patient data, returns clinical decision"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    required = ["symptoms"]
    for field in required:
        if not data.get(field):
            return jsonify({
                "success": False,
                "error": f"Missing required field: {field}"
            }), 400

    symptoms = {
        "age":             data.get("age", "unknown"),
        "gender":          data.get("gender", "unknown"),
        "symptoms":        data.get("symptoms", ""),
        "temperature":     data.get("temperature", "not recorded"),
        "heart_rate":      data.get("heart_rate", "not recorded"),
        "resp_rate":       data.get("resp_rate", "not recorded"),
        "duration":        data.get("duration", "unknown"),
        "additional_info": data.get("additional_info", ""),
    }

    try:
        eng = get_engine()
        result = eng.get_clinical_decision(symptoms)

        # Log for demo purposes (no patient data stored)
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "triage":    result["triage"],
            "model":     result["model"],
        }
        os.makedirs("app/logs", exist_ok=True)
        with open("app/logs/assessments.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return jsonify({
            "success":  True,
            "response": result["response"],
            "triage":   result["triage"],
            "sources":  result["sources"],
            "model":    result["model"],
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status":  "ok",
        "offline": True,
        "version": "1.0.0",
        "model":   "ruralmED (Gemma 4 E4B fine-tuned)"
    })

@app.route("/api/stats")
def stats():
    """Basic usage stats for the demo"""
    log_path = "app/logs/assessments.jsonl"
    total = 0
    triage_counts = {"red": 0, "yellow": 0, "green": 0}

    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total += 1
                    t = entry.get("triage", "yellow")
                    triage_counts[t] = triage_counts.get(t, 0) + 1
                except:
                    pass

    return jsonify({
        "total_assessments": total,
        "triage_distribution": triage_counts
    })

if __name__ == "__main__":
    print("=" * 50)
    print("RuralMED AI - Starting Server")
    print("=" * 50)
    print("Offline mode: YES")
    print("No internet required")
    print()
    print("Loading AI engine (first load takes ~30 seconds)...")

    # Pre-load engine before first request
    try:
        get_engine()
        print("Engine loaded successfully!")
    except Exception as e:
        print(f"Warning: Engine pre-load failed: {e}")
        print("Engine will load on first request.")

    print()
    print("Server running at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
