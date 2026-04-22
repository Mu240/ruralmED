# rag/inference.py - Final working version for Gemma 4 thinking mode

import chromadb
from chromadb.utils import embedding_functions
import requests, json

OLLAMA_URL      = "http://localhost:11434/api/generate"
PREFERRED_MODEL = "gemma4:e4b"
FALLBACK_MODEL  = "gemma4:e4b"

class RuralMEDEngine:
    def __init__(self):
        print("Initializing RuralMED Engine...")
        client = chromadb.PersistentClient(path="rag/guidelines_db")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = client.get_collection(
            "clinical_guidelines",
            embedding_function=ef
        )
        self.model = self._get_available_model()
        print(f"Using model: {self.model}")
        print(f"Guidelines loaded: {self.collection.count()} chunks")
        print("Engine ready.\n")

    def _get_available_model(self) -> str:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if PREFERRED_MODEL in models:
                return PREFERRED_MODEL
            if "ruralmED:latest" in models:
                return "ruralmED:latest"
            for m in models:
                if "ruralmed" in m.lower():
                    return m
            return FALLBACK_MODEL
        except:
            return FALLBACK_MODEL

    def retrieve_guidelines(self, query: str, n: int = 3):
        results = self.collection.query(query_texts=[query], n_results=n)
        return results["documents"][0], results["metadatas"][0]

    def get_clinical_decision(self, symptoms: dict) -> dict:
        query = (
            f"Patient: {symptoms.get('age','unknown')}, "
            f"{symptoms.get('gender','unknown')}. "
            f"Symptoms: {symptoms.get('symptoms','')}. "
            f"Temp:{symptoms.get('temperature','?')} "
            f"HR:{symptoms.get('heart_rate','?')} "
            f"RR:{symptoms.get('resp_rate','?')}. "
            f"Duration:{symptoms.get('duration','?')}."
        )

        docs, metas = self.retrieve_guidelines(query, n=3)
        context = "\n".join([
            f"[{m['source']}]: {d[:400]}"
            for d, m in zip(docs, metas)
        ])

        prompt = f"""You are RuralMED, a clinical decision support tool for trained rural health workers.

Patient presentation:
{query}

Relevant WHO/MSF guidelines:
{context}

Provide your assessment in this exact format:

TRIAGE: [RED = refer urgently / YELLOW = monitor closely / GREEN = treat at clinic]
DIAGNOSIS: [most likely condition]
TREATMENT: [specific medication name and dose]
REFERRAL: [Yes or No, with one clear reason]
FOLLOWUP: [when to reassess this patient]"""

        # Stream response — handles Gemma 4 long thinking time
        raw_response = ""
        try:
            print("Waiting for model response", end="", flush=True)
            resp = requests.post(OLLAMA_URL, json={
                "model":  self.model,
                "prompt": prompt,
                "stream": True,
            }, stream=True, timeout=600)

            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                raw_response += chunk.get("response", "")
                if len(raw_response) % 100 == 0:
                    print(".", end="", flush=True)
                if chunk.get("done"):
                    break
            print(" done!")

        except requests.exceptions.ConnectionError:
            raw_response = "ERROR: Cannot connect to Ollama. Run 'ollama serve' in a separate terminal."
        except requests.exceptions.ReadTimeout:
            raw_response = "ERROR: Timeout. Try running 'ollama run ruralmED hello' first to warm up."
        except Exception as e:
            raw_response = f"ERROR: {e}"

        # Strip Gemma 4 thinking section
        clean = raw_response

        # Method 1: ...done thinking. marker
        if "...done thinking." in clean:
            clean = clean.split("...done thinking.")[-1].strip()

        # Method 2: done thinking without dots
        elif "done thinking" in clean.lower():
            idx = clean.lower().rfind("done thinking")
            after = clean[idx + len("done thinking"):]
            # skip punctuation and whitespace
            after = after.lstrip(".\n\r ")
            if after:
                clean = after.strip()

        # Method 3: thinking process header
        elif "thinking process:" in clean.lower():
            parts = clean.lower().split("thinking process:")
            if len(parts) > 1:
                # find where the actual response starts after thinking
                last_part = clean[clean.lower().rfind("thinking process:"):]
                # look for TRIAGE: which marks real response
                if "triage:" in last_part.lower():
                    idx = last_part.lower().find("triage:")
                    clean = last_part[idx:].strip()

        # Final cleanup
        clean = clean.strip()

        # If still empty, use raw
        if not clean:
            clean = raw_response.strip()

        triage = self._parse_triage(clean)
        return {
            "response": clean,
            "triage":   triage,
            "sources":  [m["source"] for m in metas],
            "model":    self.model,
            "query":    query
        }

    def _parse_triage(self, response: str) -> str:
        r = response.upper()
        if "RED" in r and any(w in r for w in ["URGENT","REFER","HOSPITAL","IMMEDIATELY"]):
            return "red"
        elif "GREEN" in r and any(w in r for w in ["STABLE","CLINIC","MANAGE","TREAT"]):
            return "green"
        return "yellow"


if __name__ == "__main__":
    engine = RuralMEDEngine()

    print("=" * 60)
    print("TEST CASE 1: Child with fever")
    print("=" * 60)
    result = engine.get_clinical_decision({
        "age":         "4 years",
        "gender":      "male",
        "symptoms":    "fever 3 days, headache, loss of appetite",
        "temperature": "38.8C",
        "heart_rate":  "110",
        "resp_rate":   "28",
        "duration":    "3 days",
    })
    print(f"\nTriage: {result['triage'].upper()}")
    print(f"Sources: {result['sources']}")
    print(f"\nResponse:\n{result['response']}")

    print("\n" + "=" * 60)
    print("TEST CASE 2: Adult breathing difficulty")
    print("=" * 60)
    result2 = engine.get_clinical_decision({
        "age":         "35 years",
        "gender":      "female",
        "symptoms":    "difficulty breathing, cough with yellow sputum, chest pain",
        "temperature": "39.2C",
        "heart_rate":  "118",
        "resp_rate":   "34",
        "duration":    "5 days",
    })
    print(f"\nTriage: {result2['triage'].upper()}")
    print(f"\nResponse:\n{result2['response']}")

    print("\n" + "=" * 60)
    print("All tests done! Move to: python app\\app.py")