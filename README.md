# Vietnamese Legal Document Retrieval System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YuITC/Vietnamese-Legal-Doc-Retrieval)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-HF%20Hub-yellow)](https://huggingface.co/YuITC/bert-base-multilingual-cased-finetuned-VNLegalDocs)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HF%20Hub-green)](https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Doc-Retrieval-Data)
[![Docker Ready](https://img.shields.io/badge/docker-ready-lightgrey)](#-docker-deployment)


## 📌 Overview
This repository implements a high-performance semantic retrieval system that returns the most relevant Vietnamese legal documents for a free-text query.  
It fine-tunes **Sentence-BERT (m-BERT backbone)** on a curated Viet-law corpus, encodes both queries and documents into the same vector space, and performs fast ANN search via **FAISS**.

![Gradio Interface Demo](assets/gradio_demo.png)


## 🔑 Key features
- Step-by-step Jupyter notebooks (`step_01_…` → `step_04_…`) explaining every stage.
- Fine-tuned **[google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased)** into **[YuITC/bert-base-multilingual-cased-finetuned-VNLegalDocs](https://huggingface.co/YuITC/bert-base-multilingual-cased-finetuned-VNLegalDocs)**, which suitable for Vietnamese legal documents retrieval.
- FAISS GPU index for sub-second vector search on >100 k documents.
- Evaluation on **BKAI Legal Doc Retrieval** from the MTEB benchmark suite.
- Gradio web UI + Python API.
- Dockerfile for one-command GPU deployment.


## 🛠️ Installation & Usage
```bash
# Clone repository
git clone https://github.com/YuITC/Vietnamese-Legal-Doc-Retrieval.git
cd Vietnamese-Legal-Doc-Retrieval

# Create env
conda create -n legal_doc_retrieval python=3.10 -y
conda activate legal_doc_retrieval

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install faiss-gpu=1.9.0 -c pytorch -c nvidia
pip install -r requirements.txt

# Running the Application
python main.py
```

The application will start a local web server with the Gradio interface, allowing you to enter legal queries and retrieve relevant documents.


## 📂 Project Structure
```
Vietnamese-Legal-Doc-Retrieval/
├── assets/                   # Visual assets
├── cache/                    # Cached model files (BERT model)
├── data/
│   ├── original/             # Original downloaded dataset
│   │   ├── corpus.csv        # Raw corpus documents
│   │   ├── train_split.csv   # Raw training data
│   │   ├── val_split.csv     # Raw validation data
│   │   └── ...
│   ├── processed/              # Processed dataset files
│   │   ├── corpus_data.parquet # Processed corpus for embedding
│   │   ├── train_data.parquet  # Processed training data
│   │   └── test_data.parquet   # Processed test data
│   └── retrieval/            # Files for retrieval system
│       └── legal_faiss.index # FAISS index for fast vector search
├── models/                   # Trained model files
│   └── VN-legalDocs-SBERT/   # Fine-tuned BERT model for legal documents
│       ├── model.safetensors # Model weights
│       ├── config.json       # Model configuration
│       └── checkpoint-*/     # Training checkpoints
├── results/                  # Evaluation results
├── Dockerfile                # Docker configuration for deployment
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
├── settings.py               # Configuration settings
└── step_*_*.ipynb            # Jupyter notebooks for each step of the process
```


## 📊 Model Training Process
Training device configuration:
- GPU: 01 x GPU Nvidia RTX A4000 16GB, 6144 CUDA cores
- CPU: 12 Core vCPU AMD EPYC 7K62, 48GB RAM
- Training time: 3:32:33(s)

The project follows a systematic approach to build the retrieval system:
1. **Data Preparation** (`step_01_Prepare_Data.ipynb`): 
   - Processes raw legal documents
   - Creates query-document pairs for training
   - Formats data for the embedding model
2. **SBERT Fine-tuning** (`step_02_Finetune_SBERT.ipynb`):
   - Fine-tunes a multilingual BERT model with legal document pairs
   - Uses `CachedMultipleNegativesRankingLoss` for training
   - Optimizes for semantic similarity in legal context
3. **Evaluation** (`step_03_Eval_with_MTEB.ipynb`):
   - Evaluates model performance using retrieval metrics
   - Compares with baseline models
4. **Retrieval System Setup** (`step_04_Retrieval.ipynb`):
   - Creates FAISS index from document embeddings
   - Implements efficient search functionality
   - Prepares for deployment


## 🧪 Performance
The fine-tuned model was evaluated using the [MTEB benchmark](https://github.com/embeddings-benchmark/mteb) on the BKAILegalDocRetrieval dataset. Key results:

| Metric       | @k  | Pre-trained model score (%) | Fine-tuned model score (%) |
|--------------|-----|-----------------------------|-----------------------------|
| **NDCG**     | 1   | 0.007                       | 42.425                      |
|              | 5   | 0.011                       | 57.387                      |
|              | 10  | 0.023                       | 60.389                      |
|              | 20  | 0.049                       | 62.160                      |
|              | 100 | 0.147                       | 63.894                      |
| **MAP**      | 1   | 0.007                       | 40.328                      |
|              | 5   | 0.009                       | 52.297                      |
|              | 10  | 0.014                       | 53.608                      |
|              | 20  | 0.021                       | 54.136                      |
|              | 100 | 0.033                       | 54.418                      |
| **Recall**   | 1   | 0.007                       | 40.328                      |
|              | 5   | 0.017                       | 70.466                      |
|              | 10  | 0.054                       | 79.407                      |
|              | 20  | 0.157                       | 86.112                      |
|              | 100 | 0.713                       | 94.805                      |
| **Precision**| 1   | 0.007                       | 42.425                      |
|              | 5   | 0.003                       | 15.119                      |
|              | 10  | 0.005                       | 8.587                       |
|              | 20  | 0.008                       | 4.687                       |
|              | 100 | 0.007                       | 1.045                       |
| **MRR**      | 1   | 0.007                       | 42.418                      |
|              | 5   | 0.010                       | 54.337                      |
|              | 10  | 0.014                       | 55.510                      |
|              | 20  | 0.021                       | 55.956                      |
|              | 100 | 0.033                       | 56.172                      |

- **NDCG@k (Normalized Discounted Cumulative Gain)**: Measures ranking quality by evaluating the relevance of results with logarithmic position-based discounting.  
- **MAP@k (Mean Average Precision)**: Computes the average precision for each query up to rank k—precision at each relevant retrieved document—then averages across all queries.  
- **Recall@k**: The proportion of all relevant documents that are retrieved in the top k results.  
- **Precision@k**: The proportion of the top k retrieved documents that are relevant.  
- **MRR@k (Mean Reciprocal Rank)**: The average of the reciprocal of the rank position of the first relevant document across all queries. 

The model significantly outperforms baseline retrieval methods, with the main evaluation score (NDCG@10) reaching 60.4%, demonstrating strong performance on Vietnamese legal document retrieval tasks.


## 🐳 Docker Deployment
The project includes a Docker configuration for easy deployment. The Docker image is built on `continuumio/miniconda3` and includes GPU support via PyTorch CUDA and FAISS-GPU.
```bash
# Build the Docker image
docker build -t vietnamese-legal-retrieval .

# Run the container
docker run -p 7860:7860 vietnamese-legal-retrieval
```

The container:
- Uses Python 3.10 with CUDA 12.1 support
- Installs required dependencies from requirements.txt
- Exposes port 7860 for the Gradio web interface
- Sets proper environment variables for security and performance
- Runs as a non-root user for enhanced security

You can access the web interface by navigating to `http://localhost:7860` after starting the container.


## 📜 License
This project is licensed under the MIT License – feel free to modify and distribute it as needed.


## 🤝 Acknowledgments
Thanks for:
- [BKAI Legal Retrieval Dataset](https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval) for the original data
- [Sentence Transformers](https://www.sbert.net/) library for the embedding model architecture
- [Hugging Face](https://huggingface.co/) for hosting the model and dataset

If you find this project useful, consider ⭐️ starring the repository or contributing to further improvements!


## 📬 Contact
For any questions or collaboration opportunities, feel free to reach out:

📧 Email: tainguyenphu2502@gmail.com
