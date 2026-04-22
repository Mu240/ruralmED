# DAYS 9-10 - rag/build_knowledge_base.py
# Run: python rag/build_knowledge_base.py
# Builds a local offline vector database from WHO/MSF guidelines
# No internet needed after first run (embeddings cached locally)

import chromadb
from chromadb.utils import embedding_functions
import os, glob

os.makedirs("rag/guidelines_db", exist_ok=True)
os.makedirs("rag/pdfs", exist_ok=True)

print("=" * 60)
print("RuralMED - Building Clinical Knowledge Base")
print("=" * 60)

# ── Initialize local ChromaDB ─────────────────────
client = chromadb.PersistentClient(path="rag/guidelines_db")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"   # small, fast, runs fully offline
)

# Delete existing collection to rebuild fresh
try:
    client.delete_collection("clinical_guidelines")
    print("Cleared existing collection")
except:
    pass

collection = client.get_or_create_collection(
    name="clinical_guidelines",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 80:
            chunks.append(chunk.strip())
    return chunks

# ── WHO / MSF Clinical Guidelines ─────────────────
# These are based on publicly available WHO IMAI and MSF guidelines
GUIDELINES = [
    {
        "source": "WHO_fever_assessment",
        "category": "fever",
        "text": """Fever Assessment Protocol (WHO Primary Care Guidelines):
        Temperature above 37.5C axillary or 38.0C rectal is defined as fever.
        High fever above 39C requires urgent assessment.
        
        Step 1 - Check for danger signs immediately:
        - Altered consciousness or confusion
        - Inability to drink or breastfeed in children
        - Repeated vomiting preventing oral medication
        - Convulsions in the past 24 hours
        - Prostration or extreme weakness
        - Severe respiratory distress
        If ANY danger sign present: give pre-referral treatment and refer URGENTLY.
        
        Step 2 - Assess for malaria in endemic areas:
        Perform Rapid Diagnostic Test (RDT) for all patients with fever.
        If RDT positive: treat for malaria per protocol.
        If RDT negative: look for other cause of fever.
        
        Step 3 - Assess for pneumonia:
        Count respiratory rate for full 60 seconds.
        Check for chest indrawing.
        Listen for stridor.
        
        Treatment: Paracetamol 15mg/kg every 6 hours for fever above 38.5C.
        Reassess in 24-48 hours. If fever persists beyond 3 days: refer."""
    },
    {
        "source": "WHO_malaria_treatment",
        "category": "malaria",
        "text": """Malaria Treatment Protocol (WHO 2023 Guidelines):
        
        DIAGNOSIS: Always confirm with RDT before treating. Do not treat presumptively.
        
        UNCOMPLICATED MALARIA (P. falciparum):
        First line: Artemether-Lumefantrine (AL) tablets
        - Under 5kg: not recommended
        - 5-14kg: 1 tablet twice daily for 3 days
        - 15-24kg: 2 tablets twice daily for 3 days
        - 25-34kg: 3 tablets twice daily for 3 days
        - Over 35kg: 4 tablets twice daily for 3 days
        Give with fatty food or milk. Complete full 3-day course.
        
        SEVERE MALARIA SIGNS (refer urgently if any present):
        - Prostration or inability to sit/stand
        - Impaired consciousness (any level)
        - Multiple convulsions (2+ in 24 hours)
        - Respiratory distress or deep breathing
        - Circulatory collapse or shock
        - Abnormal bleeding
        - Jaundice with other organ dysfunction
        - Haemoglobinuria (black/dark urine)
        
        PRE-REFERRAL TREATMENT FOR SEVERE MALARIA:
        Artesunate rectal (if available): 10mg/kg as single dose before transfer.
        OR Quinine IM: 10mg/kg if artesunate not available.
        Start IV fluids. Treat hypoglycemia with glucose."""
    },
    {
        "source": "WHO_pneumonia_children",
        "category": "respiratory",
        "text": """Pneumonia Assessment and Treatment in Children (WHO IMAI):
        
        RESPIRATORY RATE THRESHOLDS (count for full 60 seconds):
        - Under 2 months: Fast breathing = 60 or more per minute
        - 2-11 months: Fast breathing = 50 or more per minute
        - 1-5 years: Fast breathing = 40 or more per minute
        - Over 5 years: Fast breathing = 30 or more per minute
        
        CLASSIFICATION:
        Fast breathing only = Non-severe pneumonia → treat at clinic
        Chest indrawing = Severe pneumonia → refer to hospital
        Danger signs present = Very severe pneumonia → refer URGENTLY
        
        TREATMENT NON-SEVERE PNEUMONIA:
        Amoxicillin oral: 40-45mg/kg/day in 2 divided doses for 5 days
        - Under 5kg: 125mg twice daily
        - 5-9kg: 250mg twice daily  
        - 10-19kg: 500mg twice daily
        - 20-29kg: 750mg twice daily
        - 30-39kg: 1000mg twice daily
        
        Reassess after 2 days. If not improving → refer.
        If stridor at rest, unable to drink, cyanosis → refer immediately."""
    },
    {
        "source": "WHO_diarrhea_dehydration",
        "category": "diarrhea",
        "text": """Diarrhea and Dehydration Management (WHO Guidelines):
        
        ASSESS DEHYDRATION:
        
        No dehydration: Alert, drinks normally, normal skin turgor, normal eyes
        Treatment: ORS at home after each loose stool
        - Under 2 years: 50-100ml ORS after each stool
        - 2-10 years: 100-200ml ORS after each stool
        - Over 10 years: as much as wanted
        Continue feeding. Zinc: 20mg/day for 10-14 days (10mg if under 6 months)
        
        Some dehydration: Restless/irritable, drinks eagerly, sunken eyes, slow skin turgor
        Treatment: Give 75ml/kg ORS over 4 hours in clinic. Reassess every hour.
        If improving after 4 hours → send home with ORS and zinc.
        
        Severe dehydration: Lethargic or unconscious, unable to drink, very sunken eyes, very slow skin turgor
        Treatment: REFER URGENTLY. Give IV Ringer Lactate 100ml/kg
        - Under 12 months: 30ml/kg over 1 hour, then 70ml/kg over 5 hours
        - Over 12 months: 30ml/kg over 30 minutes, then 70ml/kg over 2.5 hours
        
        BLOODY DIARRHEA (Dysentery):
        Treat with Ciprofloxacin 15mg/kg twice daily for 3 days.
        Refer if not improving within 2 days."""
    },
    {
        "source": "WHO_malnutrition",
        "category": "nutrition",
        "text": """Malnutrition Assessment and Management (WHO):
        
        ASSESS NUTRITIONAL STATUS:
        MUAC (Mid-Upper Arm Circumference):
        - Red: Under 11.5cm = Severe Acute Malnutrition (SAM)
        - Yellow: 11.5-12.5cm = Moderate Acute Malnutrition (MAM)
        - Green: Over 12.5cm = Normal
        
        Weight-for-Height Z-score:
        - Below -3 SD = SAM
        - -3 to -2 SD = MAM
        
        SEVERE ACUTE MALNUTRITION (SAM):
        Refer to hospital if ANY of: bilateral edema, medical complications, poor appetite, MUAC under 11.5cm with complications.
        
        If no complications: Outpatient Therapeutic Program (OTP)
        RUTF (Ready-to-Use Therapeutic Food): 200kcal/kg/day
        - Under 4kg: 3 sachets per day
        - 4-6.9kg: 4 sachets per day
        - 7-9.9kg: 5 sachets per day
        - 10-14.9kg: 6 sachets per day
        Weekly follow-up. Discharge when MUAC over 12.5cm for 2 consecutive weeks."""
    },
    {
        "source": "WHO_antenatal_care",
        "category": "maternal",
        "text": """Antenatal Care Red Flags - Refer Immediately (WHO):
        
        Any of the following require URGENT referral:
        - Vaginal bleeding at any stage of pregnancy
        - Severe headache with visual disturbance
        - Blood pressure 140/90 or above (pre-eclampsia)
        - Severe abdominal pain
        - Fever above 38C in pregnancy
        - Convulsions or seizures (eclampsia)
        - Difficulty breathing
        - Baby not moving normally after 28 weeks
        - Signs of preterm labor before 37 weeks
        
        PRE-ECLAMPSIA MANAGEMENT BEFORE REFERRAL:
        If BP 140/90 or above with headache/visual changes:
        Give Magnesium Sulfate 4g IV over 20 minutes as loading dose.
        Maintain on 1g/hour. Give Nifedipine 10mg orally if BP above 160/110.
        Transfer immediately with IV in place."""
    },
    {
        "source": "WHO_triage_system",
        "category": "triage",
        "text": """Primary Care Triage System (Emergency Triage Assessment Treatment):
        
        RED - EMERGENCY (treat immediately, do not wait):
        - Airway obstruction or severe respiratory distress
        - Shock (cold extremities, weak fast pulse, low BP, altered consciousness)
        - Severe dehydration with altered consciousness
        - Convulsions actively occurring
        - Unconscious or unresponsive patient
        - Severe trauma with major bleeding
        
        YELLOW - URGENT (assess within 30 minutes):
        - High fever with any warning sign
        - Moderate dehydration
        - Severe pain
        - History of convulsions (now stable)
        - Severe malnutrition
        - Unable to drink or feed
        
        GREEN - NON-URGENT (can wait, treat and send home):
        - Mild fever without danger signs
        - Non-severe diarrhea without dehydration
        - Upper respiratory infection
        - Minor injuries
        - Routine follow-up"""
    },
    {
        "source": "WHO_newborn_danger_signs",
        "category": "newborn",
        "text": """Newborn Danger Signs - Refer Immediately (WHO):
        
        The following require IMMEDIATE referral of newborn to hospital:
        - Not breathing or breathing with difficulty
        - Convulsions
        - Unable to feed or stopped feeding
        - Very cold (temperature below 35.5C)
        - Very hot (temperature above 38C)
        - Yellow skin on face or chest (jaundice in first 24 hours)
        - Umbilical redness spreading to skin
        - Bleeding from any site
        - Floppy or lethargic
        - Less than 12 movements in 12 hours after day 3
        
        KANGAROO MOTHER CARE for low birth weight (under 2kg):
        Skin-to-skin contact 24 hours a day. Mother's chest between breasts.
        Exclusive breastfeeding every 2-3 hours.
        Keep warm. Monitor temperature 4 times daily.
        Weekly weight check. Discharge when weight over 2.5kg."""
    },
]

# ── Add guidelines to ChromaDB ────────────────────
print(f"\nAdding {len(GUIDELINES)} clinical guideline documents...")
all_chunks = []
all_ids    = []
all_metas  = []

for guide in GUIDELINES:
    chunks = chunk_text(guide["text"])
    for j, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{guide['source']}_chunk_{j}")
        all_metas.append({
            "source":   guide["source"],
            "category": guide["category"]
        })
    print(f"  {guide['source']}: {len(chunks)} chunks")

# Add in batches to avoid memory issues
batch_size = 50
for i in range(0, len(all_chunks), batch_size):
    collection.add(
        documents=all_chunks[i:i+batch_size],
        ids=all_ids[i:i+batch_size],
        metadatas=all_metas[i:i+batch_size]
    )

# ── Process any PDFs in rag/pdfs/ ─────────────────
pdf_files = glob.glob("rag/pdfs/*.pdf")
if pdf_files:
    print(f"\nProcessing {len(pdf_files)} PDF files...")
    try:
        from PyPDF2 import PdfReader
        for pdf_path in pdf_files:
            name = os.path.basename(pdf_path).replace(".pdf", "")
            reader = PdfReader(pdf_path)
            text = " ".join(
                page.extract_text() or "" for page in reader.pages
            )
            chunks = chunk_text(text)
            collection.add(
                documents=chunks,
                ids=[f"{name}_pdf_chunk_{j}" for j in range(len(chunks))],
                metadatas=[{"source": name, "category": "pdf"} for _ in chunks]
            )
            print(f"  Added PDF: {name} ({len(chunks)} chunks)")
    except ImportError:
        print("  PyPDF2 not installed, skipping PDFs")

# ── Test the knowledge base ───────────────────────
print("\n" + "=" * 60)
print(f"Total chunks in database: {collection.count()}")
print("\nTesting retrieval...")

test_query = "child with high fever and rapid breathing"
results = collection.query(query_texts=[test_query], n_results=3)

print(f"\nQuery: '{test_query}'")
print("Top 3 relevant guidelines:")
for i, (doc, meta) in enumerate(zip(
    results["documents"][0], results["metadatas"][0]
)):
    print(f"\n  [{i+1}] Source: {meta['source']}")
    print(f"       {doc[:150]}...")


