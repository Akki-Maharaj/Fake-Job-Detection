# Fake Job Postings Detection using DistilBERT

An NLP classifier that detects fraudulent job postings using fine-tuned DistilBERT, with full evaluation metrics and TensorBoard visualization.

---

## Project Structure

```
fake_job_detection/
├── data/                        # Dataset and generated plots
│   └── fake_job_postings.csv    # Download from Kaggle (see Setup)
├── models/
│   ├── tokenizer/               # Saved DistilBERT tokenizer
│   └── best_model/              # Best checkpoint (saved by val F1)
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_Tokenization.ipynb    # Tokenization walkthrough
│   ├── 03_Training.ipynb        # Fine-tuning DistilBERT
│   └── 04_Evaluation.ipynb      # Metrics, confusion matrix, misclassifications
├── tensorboard_logs/            # TensorBoard event files
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd fake_job_detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    → https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
#    → Place fake_job_postings.csv in the data/ folder

# 5. Run notebooks in order
jupyter lab
```

---

## Run Notebooks in Order

| Notebook | What it does |
|----------|-------------|
| `01_EDA.ipynb` | Explores the dataset, class imbalance, text lengths |
| `02_Tokenization.ipynb` | Explains and demonstrates DistilBERT tokenization |
| `03_Training.ipynb` | Fine-tunes DistilBERT with weighted loss for imbalance |
| `04_Evaluation.ipynb` | Full evaluation: F1, AUC-ROC, confusion matrix, PR curve |

---

## View TensorBoard

```bash
tensorboard --logdir tensorboard_logs
# Open http://localhost:6006
```

---

## Dataset

**Source:** [Kaggle — Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

- ~17,880 job postings
- ~17,000 real (95%) / ~800 fake (5%) — heavily imbalanced
- Features: title, company_profile, description, requirements, benefits, location, and more

---

## Key Design Decisions

### 1. Class Imbalance → Weighted Cross-Entropy Loss
The dataset is ~95% real / ~5% fake. A naive model predicting "real" for everything achieves 95% accuracy but detects zero fraud. We use **weighted cross-entropy loss** to penalize false negatives on the minority class more heavily.

### 2. Multiple Text Fields Combined
Rather than using only the job description, we concatenate: title + company profile + description + requirements + benefits. This gives the model more signal.

### 3. Stratified Train/Val/Test Split
With only ~800 fake examples, a random split could produce validation sets with too few fakes to measure performance reliably. Stratified splitting preserves the class ratio in every split.

### 4. Model Saved on F1, Not Accuracy
The best checkpoint is chosen based on **macro F1** on the validation set — not accuracy — since accuracy is misleading for imbalanced data.

### 5. Linear LR Warmup + Decay
Standard for transformer fine-tuning. The first 10% of training steps linearly increase the LR from 0 to peak, then decay. This prevents early large updates from destabilizing pre-trained weights.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | — |
| Macro F1 | — |
| Fake Recall | — |
| AUC-ROC | — |

*(Fill in after running)*

---

## Tech Stack

- Python 3.10
- PyTorch 2.1
- Hugging Face Transformers (DistilBERT)
- scikit-learn
- TensorBoard
- pandas, numpy, matplotlib, seaborn
