# 📜 METHODOLOGY : Sapt Sindhu Synthetic Data Architecture

## 1. How It Works: The Generation Pipeline
The project utilizes a **Constraint-Satisfied Generation (CSG)** pipeline designed to create historically accurate instruction-response pairs for the Punjab/Sindh region (1500–1600 AD). The process is automated across three distinct phases:

### Phase 1: Combinatorial Queue Construction
Instead of relying on random prompting, the system deterministically pre-calculates valid scenarios.
* **Input:** Uses `roles.json` (occupations), `historical_figures.json` (Gurus, Sufis), and `global_config.json`.
* **Logic:** The `queue_builder.py` script creates a Cartesian product of **Roles × Settings × Historical Anchors × Teachings**. It applies logic filters (e.g., ensuring a "Sufi Novice" role is only matched with Islamic/Sufi figures) to generate a `queue.json` file containing thousands of unique, valid prompt configurations.

### Phase 2: The Generation Engine
The `generator.py` script acts as the core engine that processes the queue and interacts with the LLM (Gemini 2.5 Flash).
* **Dynamic Prompting:** It constructs system prompts that enforce strict constraints, such as banning modern technology or New World crops (e.g., potatoes, corn) defined in `allowed_forbidden_list.json`.
* **Automated Validation:** The script automatically validates outputs. It rejects stories that are too short (<160 words) or do not end with a complete sentence.
* **State Management:** It uses `progress.json` to track SHA-256 hashes of generated content, preventing duplicates and allowing the script to resume exactly where it left off in case of a crash.

### Phase 3: System Layer & Formatting
* **System Layer:** A `system_layer.json` file defines narrative constraints (e.g., "no modern political ideology") and lexical ownership rules (e.g., specific terms for Sufi Islam vs. Early Sikhism) to maintain tone and authority.
* **Formatting:** Finally, `jsonl_conversion.py` converts the raw outputs into standard JSONL format (`{"prompt": "...", "response": "..."}`) suitable for fine-tuning models like Llama 3 or Mistral.

---

## 2. Why This is a "Good" Method (Advantages)

This architecture addresses common pitfalls in synthetic data generation (repetitiveness, hallucinations, and lack of diversity) through the following mechanisms:

### A. Combinatorial Diversity vs. Randomness
* **The Problem:** Asking an AI to "write 1000 random stories" typically results in mode collapse, where the model repeats common tropes (e.g., "A farmer in a field") and ignores niche scenarios.
* **The Solution:** By programmatically generating the queue in Phase 1, this method guarantees coverage of rare roles (e.g., *Sand Dredgers*, *Araghat Operators*) and specific historical settings that generic prompting would miss.

### B. Hallucination Control (The "History Guard")
* **The Problem:** LLMs often introduce anachronisms (e.g., characters drinking tea or eating corn in 1550 AD India, despite these arriving later).
* **The Solution:** The pipeline feeds explicit "Allowed/Forbidden" lists directly into the system prompt. The `system_layer.json` defines severe violations (like "caste violation" or "hierarchy inversion") which trigger hard rejects, significantly reducing historical errors.

### C. State-Aware Resilience
* **The Problem:** Generating large datasets requires thousands of API calls. Network failures or rate limits can crash scripts, leading to data loss or duplicates.
* **The Solution:** The `progress.json` system records every successful generation hash. If the script crashes, it resumes instantly without regenerating a single token, saving time and API costs.

### D. Modular Logic
* **The Solution:** The architecture separates data (JSON files) from logic (Python scripts). This means a researcher can add a new historical figure or update a banned crop list without rewriting the code, making the system highly maintainable and scalable.
