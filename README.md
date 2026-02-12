# Operation Ledger Mind: Financial RAG vs. Fine-Tuning Arena

## 1. Executive Summary
**Operation Ledger Mind** is a specialized engineering project designed to evaluate and compare two advanced architectures for financial Question Answering (QA):
1.  **Retrieval-Augmented Generation (RAG)**: A "Librarian" system using Hybrid Search (Vector + BM25) and Cross-Encoder Reranking.
2.  **Fine-Tuned LLM**: An "Intern" model (Llama-3-8B) fine-tuned on specific domain data using QLoRA.

The project focuses on extracting high-fidelity insights from **Uber's 2024 Annual Report (10-K)**, rigorously testing both systems on "Hard Facts", "Strategic Summaries", and "Stylistic" questions.

---

## 2. Project Architecture

### The "Intern" (Fine-Tuning Track)
*   **Model**: Unsloth/Llama-3-8B-Instruct (4-bit quantized)
*   **Method**: QLoRA (Low-Rank Adaptation)
*   **Training Data**: 600+ synthetic QA pairs generated from the 10-K report.
*   **Goal**: To internalize financial knowledge directly into the model's weights.

### The "Librarian" (RAG Track)
*   **Vector Database**: Weaviate (running via Docker)
*   **Retrieval**: Hybrid Search (Sparse BM25 + Dense Vectors)
*   **Refinement**: Cross-Encoder Reranking (MS-MARCO) + Reciprocal Rank Fusion (RRF).
*   **Generation**: GPT-4o-mini (context-aware generation).
*   **Goal**: To retrieve precise evidence and generate grounded answers without hallucination.

---

## 3. Repository Structure

```text
Operation Ledger Mind/
├── data/
│   ├── raw/                  # Source PDF (Uber 10-K)
│   ├── processed/            # Generated JSONL datasets (train/test)
│   └── generated/            # Evaluation results (CSVs)
├── notebooks/
│   ├── 01_data_factory.ipynb       # Data ingestion, chunking, and synthetic generation
│   ├── 02_finetuning_intern.ipynb  # Fine-tuning Llama-3 using SFTTrainer
│   ├── 03_rag_librarian.ipynb      # RAG pipeline setup (Weaviate + Search)
│   └── 04_evaluation_arena.ipynb   # Head-to-head evaluation (ROUGE + LLM Judge)
├── src/                      # Source code and helper scripts
├── utils/                    # Utility modules (logging, JSON handling, etc.)
├── docker-compose.yml        # Weaviate configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 4. Installation & Setup

### Prerequisites
*   Python 3.10+
*   Docker Desktop (for Weaviate)
*   GPU (Optional but recommended for Fine-tuning, e.g., NVIDIA T4/A10G)

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/operation-ledger-mind.git
    cd operation-ledger-mind
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables**
    Create a `.env` file in the root directory:
    ```ini
    OPENAI_API_KEY=sk-...
    WANDB_API_KEY=... (Optional)
    ```

5.  **Start Vector Database**
    ```bash
    docker-compose up -d
    ```

---

## 5. Usage Workflow

Follow the notebooks in order:

1.  **`01_data_factory.ipynb`**:
    *   Ingests `Uber_annual_report_2024.pdf`.
    *   Chunks text and uses OpenAI to generate a synthetic dataset.
    *   **Output**: `train.jsonl` and `golden_test_set.jsonl`.

2.  **`02_finetuning_intern.ipynb`**:
    *   Loads the synthetic training data.
    *   Fine-tunes the Llama-3 model using QLoRA.
    *   Saves adapters to `outputs/checkpoint-100`.

3.  **`03_rag_librarian.ipynb`**:
    *   Indexes the PDF chunks into Weaviate.
    *   Configures Hybrid Search and Reranking.
    *   Tests the retrieval pipeline.

4.  **`04_evaluation_arena.ipynb`**:
    *   Runs the "Golden Test Set" against both the Fine-Tuned Model and the RAG System.
    *   Scores results using **ROUGE-L** (text overlap) and **LLM-as-a-Judge** (faithfulness).
    *   Generates the final comparison report.

---

## 6. Key Findings
The evaluation revealed that for strict financial domains:
*   **Winner**: **RAG System ("The Librarian")**
    *   **Accuracy**: High (LLM Judge ~4.5/5). Grounded in retrieved facts.
    *   **Cost**: Low (~$53/mo for 150k queries).
*   **Fine-Tuning**: Failed on "Hard Facts" (hallucinated numbers) but performed well on stylistic/tonal tasks.

## 7. License
This project is for educational purposes as part of the ZUU Crew AI Engineering Bootcamp.
