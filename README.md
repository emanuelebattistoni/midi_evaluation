## Audio Evaluation Toolkit: CLAP & FAD

This repository includes two standalone, plug-and-play evaluation modules designed to measure the quality and semantic alignment of generated audio. They can be easily integrated into *any* text-to-audio or generative audio pipeline.

### Requirements
Before using these evaluation modules, ensure you have the necessary dependencies installed:

```bash
pip install torch librosa fadtk laion_clap
```

---

### 1. CLAP Evaluator (Text-to-Audio Alignment)
**Module:** `evaluate_clap.py`

**What it does:** Measures how accurately an audio file reflects a given text prompt. It uses the LAION-CLAP model to extract embeddings from both the text and the audio, calculating a cosine similarity score. Higher scores indicate a stronger semantic match between the prompt and the audio.

**Key Features:**
* **Framework Agnostic:** Can be imported and used in any Python project.
* **Fail-Safe Design:** Gracefully handles missing dependencies and suppresses verbose initialization logs.
* **Standardized Processing:** Automatically resamples input audio to 48kHz to meet CLAP tensor requirements.

**Usage Example:**
Simply import the functions into your Python script:

```python
from evaluate_clap import init_clap, evaluate_audio_clap

# 1. Load the CLAP model into memory (run this once)
init_clap()

# 2. Evaluate any audio file against a text description
score = evaluate_audio_clap(
    text_prompt="A heavy rock drum beat with double kick",
    audio_path="/path/to/generated_audio.mp3"
)
print(f"Alignment Score: {score:.4f}")
```

---

### 2. FAD Evaluator (Fréchet Audio Distance)
**Script:** `evaluate_fad.py`

**What it does:** Calculates the Fréchet Audio Distance (FAD) to measure the acoustic gap between a baseline dataset (real, reference audio) and an evaluation dataset (AI-generated audio). A lower FAD score indicates that the generated audio closely matches the acoustic characteristics and diversity of the real data.

**Key Features:**
* **Standalone CLI:** Operates independently of your generation code.
* **Auto-Filtering Magic:** Recursively scans the target directory, isolates only `.mp3` files into a secure temporary environment, and prevents crashes caused by mixed-file directories (e.g., automatically ignoring `.json` or `.mid` files).
* **Multi-Model Support:** Powered by `fadtk`, it uses `clap-2023` embeddings by default but supports others like `vggish` or `encodec-emb`.

**Usage Example:**
Run the script directly from your terminal, pointing it to any two directories containing audio files:

```bash
python evaluate_fad.py \
    --reference /path/to/real_dataset_folder \
    --generated /path/to/ai_dataset_folder \
    --model clap-2023
```

The script will output the calculation process to the console and automatically save a detailed `REPORT_FAD.txt` in your generated dataset folder.