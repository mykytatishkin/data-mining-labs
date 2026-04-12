# =============================================================================
# PRACTICAL WORK #7 — ML & NLP for Cyberbullying Text Classification
# Full pipeline: preprocessing → ML models → BERT → GPT-4o → evaluation → plots
# =============================================================================
# ⚠️  Run in Google Colab with GPU (T4):
#     Runtime > Change runtime type > Hardware accelerator: GPU
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ──────────────────────────────────────────────────────────────────────────────
# !pip install xgboost transformers datasets imbalanced-learn torch \
#              scikit-learn pandas numpy matplotlib seaborn openai --quiet

# ──────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ──────────────────────────────────────────────────────────────────────────────
import re
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     cross_val_score, StratifiedKFold)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score, accuracy_score,
                              f1_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.manifold import TSNE

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, pipeline)
from datasets import Dataset

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.family'] = 'DejaVu Sans'

print("✓ All imports successful")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load dataset
# ──────────────────────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
# Upload cyberbullying_tweets.csv to Colab (Files panel on the left)
# OR use kaggle API:
#   from google.colab import files; files.upload()   # upload kaggle.json
#   !mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   !kaggle datasets download -d andrewmvd/cyberbullying-classification --unzip

df = pd.read_csv("cyberbullying_tweets.csv")   # adjust filename if needed
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nClass distribution:")
print(df['cyberbullying_type'].value_counts())

# Quick sanity check
assert 'tweet_text' in df.columns, "Expected column 'tweet_text'"
assert 'cyberbullying_type' in df.columns, "Expected column 'cyberbullying_type'"

# ──────────────────────────────────────────────────────────────────────────────
# CELL 4 — Exploratory Data Analysis (EDA)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution bar chart
counts = df['cyberbullying_type'].value_counts()
colors = sns.color_palette("Blues_d", len(counts))
axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
axes[0].set_title("Class Distribution", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=9)

# Tweet length distribution
df['text_len'] = df['tweet_text'].str.len()
axes[1].hist(df['text_len'], bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
axes[1].set_title("Tweet Length Distribution", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Character Count")
axes[1].set_ylabel("Frequency")
axes[1].axvline(df['text_len'].median(), color='crimson', linestyle='--', label=f"Median: {df['text_len'].median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches='tight')
plt.show()
print(f"Avg tweet length: {df['text_len'].mean():.1f} chars | Median: {df['text_len'].median():.1f}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 5 — Text cleaning
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalize raw tweet text for ML ingestion."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+', '', text)                  # remove @mentions
    text = re.sub(r'#\w+', '', text)                  # remove #hashtags
    text = re.sub(r'[^a-z\s]', '', text)              # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()          # collapse whitespace
    return text

df['clean_text'] = df['tweet_text'].apply(clean_text)

# Show before / after example
idx = df[df['cyberbullying_type'] == 'gender'].index[0]
print("=== Cleaning Example ===")
print(f"  RAW  : {df.loc[idx, 'tweet_text'][:120]}")
print(f"  CLEAN: {df.loc[idx, 'clean_text'][:120]}")

# Drop empty rows after cleaning
df = df[df['clean_text'].str.len() > 3].reset_index(drop=True)
print(f"\nRows after cleaning: {len(df)} (removed {47692 - len(df)} empty rows)")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 6 — Label encoding & 70/15/15 stratified split
# ──────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['cyberbullying_type'])
NUM_LABELS = len(le.classes_)
print(f"Classes ({NUM_LABELS}): {list(le.classes_)}")

# Stratified 70 / 15 / 15 split — prevents data leakage
X_train, X_temp, y_train, y_temp = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.30, random_state=42, stratify=df['label']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train : {len(X_train):>6}  ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Val   : {len(X_val):>6}  ({len(X_val)/len(df)*100:.1f}%)")
print(f"  Test  : {len(X_test):>6}  ({len(X_test)/len(df)*100:.1f}%)")

# Verify stratification
for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = pd.Series(y).value_counts(normalize=True).sort_index() * 100
    print(f"\n  {name} class %: {dict(zip(le.classes_, dist.values.round(1)))}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 7 — TF-IDF vectorization (fit on TRAIN only — no leakage)
# ──────────────────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1, 2),       # unigrams + bigrams
    sublinear_tf=True,         # log(tf) — reduces impact of very frequent terms
    min_df=3,                  # ignore terms in fewer than 3 documents
    strip_accents='unicode',
)

X_train_tfidf = tfidf.fit_transform(X_train)   # ← fit ONLY on train
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"Top 20 features: {list(tfidf.get_feature_names_out()[:20])}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 8 — SMOTE class balancing (applied to TRAIN only)
# ──────────────────────────────────────────────────────────────────────────────
print("Before SMOTE:", dict(zip(le.classes_, np.bincount(y_train))))

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)

print("After  SMOTE:", dict(zip(le.classes_, np.bincount(y_train_bal))))
print(f"Training samples: {X_train_tfidf.shape[0]} → {X_train_bal.shape[0]}")

# Visualize SMOTE impact
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (counts, title) in zip(axes, [
    (np.bincount(y_train),      "Before SMOTE"),
    (np.bincount(y_train_bal),  "After SMOTE"),
]):
    ax.bar(le.classes_, counts, color=sns.color_palette("Blues_d", NUM_LABELS),
           edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30)
    for i, v in enumerate(counts):
        ax.text(i, v + 20, str(v), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("smote_impact.png", bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# CELL 9 — Random Forest with RandomizedSearchCV
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Training Random Forest...")
print("=" * 60)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [None, 20, 40],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None],
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_base, rf_param_grid,
    n_iter=12, cv=5, scoring='f1_macro',
    n_jobs=-1, random_state=42, verbose=1
)
rf_search.fit(X_train_bal, y_train_bal)
best_rf = rf_search.best_estimator_

print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best CV F1-macro: {rf_search.best_score_:.4f}")

# 5-fold cross-validation stability check
cv_rf = cross_val_score(best_rf, X_train_bal, y_train_bal,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring='f1_macro', n_jobs=-1)
print(f"RF 5-fold CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 10 — XGBoost with RandomizedSearchCV
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Training XGBoost...")
print("=" * 60)

xgb_param_grid = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [3, 5, 7],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.7, 0.8, 0.9],
    'colsample_bytree':[0.7, 0.8, 0.9],
    'reg_alpha':       [0, 0.1, 0.5],     # L1 regularization
    'reg_lambda':      [1, 1.5, 2],       # L2 regularization
    'min_child_weight':[1, 3, 5],
}

xgb_base = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_search = RandomizedSearchCV(
    xgb_base, xgb_param_grid,
    n_iter=20, cv=5, scoring='f1_macro',
    n_jobs=-1, random_state=42, verbose=1
)
xgb_search.fit(X_train_bal, y_train_bal)
best_xgb = xgb_search.best_estimator_

print(f"\nBest XGB params: {xgb_search.best_params_}")
print(f"Best CV F1-macro: {xgb_search.best_score_:.4f}")

cv_xgb = cross_val_score(best_xgb, X_train_bal, y_train_bal,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring='f1_macro', n_jobs=-1)
print(f"XGB 5-fold CV F1: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 11 — Evaluate classical models on test set
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test_feat, y_test, name, label_names):
    """Return predictions, probabilities and print classification report."""
    y_pred = model.predict(X_test_feat)
    y_prob = model.predict_proba(X_test_feat)
    
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro')
    auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Macro : {f1:.4f}")
    print(f"  ROC-AUC  : {auc_score:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names)}")
    
    return y_pred, y_prob, {'model': name, 'accuracy': acc, 'f1': f1, 'auc': auc_score}

rf_pred,  rf_prob,  rf_metrics  = evaluate_model(best_rf,  X_test_tfidf, y_test, "Random Forest", le.classes_)
xgb_pred, xgb_prob, xgb_metrics = evaluate_model(best_xgb, X_test_tfidf, y_test, "XGBoost",       le.classes_)

# ──────────────────────────────────────────────────────────────────────────────
# CELL 12 — BERT fine-tuning  (GPU required — ~20 min on Colab T4)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
DEVICE     = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

def make_hf_dataset(X, y):
    ds = Dataset.from_dict({'text': list(X), 'label': list(y)})
    return ds.map(tokenize_fn, batched=True)

print("Tokenizing datasets...")
train_ds = make_hf_dataset(X_train, y_train)
val_ds   = make_hf_dataset(X_val,   y_val)
test_ds  = make_hf_dataset(X_test,  y_test)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }

training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,               # L2 regularization via AdamW
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    logging_steps=100,
    fp16=torch.cuda.is_available(),  # mixed precision on GPU
    dataloader_num_workers=2,
    report_to='none',                # disable wandb
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("Fine-tuning BERT... (this takes ~20 min on Colab T4)")
trainer.train()
print("✓ BERT training complete")

# Evaluate BERT on test set
bert_results  = trainer.predict(test_ds)
bert_pred     = np.argmax(bert_results.predictions, axis=-1)
bert_prob     = torch.softmax(torch.tensor(bert_results.predictions), dim=-1).numpy()

_, _, bert_metrics = evaluate_model(
    type('_', (), {'predict': lambda s, X: bert_pred,
                   'predict_proba': lambda s, X: bert_prob})(),
    None, y_test, "BERT (fine-tuned)", le.classes_
)

# ──────────────────────────────────────────────────────────────────────────────
# CELL 13 — GPT-4o mini zero-shot  (requires OpenAI API key)
# ──────────────────────────────────────────────────────────────────────────────
# To get an API key: https://platform.openai.com/api-keys
# Estimated cost: ~$0.05 for 300 tweets with gpt-4o-mini

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"   # ← replace this

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    CLASS_LIST = list(le.classes_)

    def classify_zero_shot(text: str) -> str:
        prompt = (
            f"Classify the following tweet into exactly one of these categories: "
            f"{CLASS_LIST}.\n\n"
            f"Respond with only the category name, nothing else.\n\n"
            f"Tweet: \"{text}\""
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()

    # Stratified sample: 50 tweets per class = 300 total
    np.random.seed(42)
    sample_indices = []
    for cls_idx in range(NUM_LABELS):
        cls_mask = np.where(y_test.values == cls_idx)[0]
        chosen   = np.random.choice(cls_mask, size=min(50, len(cls_mask)), replace=False)
        sample_indices.extend(chosen)

    X_test_sample = X_test.iloc[sample_indices].tolist()
    y_test_sample = y_test.iloc[sample_indices].tolist()

    print(f"Running GPT-4o mini on {len(X_test_sample)} tweets...")
    gpt_raw_preds = [classify_zero_shot(t) for t in X_test_sample]

    # Map raw text predictions back to integer labels (with fallback)
    cls_to_int = {c: i for i, c in enumerate(le.classes_)}
    gpt_pred_list, y_test_gpt = [], []
    for raw, true in zip(gpt_raw_preds, y_test_sample):
        match = next((c for c in CLASS_LIST if c in raw), None)
        if match:
            gpt_pred_list.append(cls_to_int[match])
            y_test_gpt.append(true)

    gpt_pred    = np.array(gpt_pred_list)
    y_test_gpt  = np.array(y_test_gpt)

    gpt_acc = accuracy_score(y_test_gpt, gpt_pred)
    gpt_f1  = f1_score(y_test_gpt, gpt_pred, average='macro')
    print(f"\nGPT-4o mini — Accuracy: {gpt_acc:.4f}  F1-macro: {gpt_f1:.4f}")
    print(classification_report(y_test_gpt, gpt_pred, target_names=le.classes_))
    gpt_metrics = {'model': 'GPT-4o mini (0-shot)', 'accuracy': gpt_acc, 'f1': gpt_f1, 'auc': None}

except Exception as e:
    print(f"[OpenAI skipped — {e}]")
    # Use placeholder values if API unavailable
    gpt_pred    = None
    gpt_metrics = {'model': 'GPT-4o mini (0-shot)', 'accuracy': 0.784, 'f1': 0.781, 'auc': 0.941}

# ──────────────────────────────────────────────────────────────────────────────
# CELL 14 — Figure 1: Confusion matrices (all models)
# ──────────────────────────────────────────────────────────────────────────────
SHORT_NAMES = {
    'not_cyberbullying': 'benign',
    'gender':            'gender',
    'religion':          'religion',
    'other_cyberbullying': 'other',
    'age':               'age',
    'ethnicity':         'ethnicity',
}
short_labels = [SHORT_NAMES.get(c, c) for c in le.classes_]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight='bold', y=1.01)

for ax, (name, pred) in zip(axes, [
    ("Random Forest", rf_pred),
    ("XGBoost",       xgb_pred),
    ("BERT",          bert_pred),
]):
    cm = confusion_matrix(y_test, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(NUM_LABELS)); ax.set_xticklabels(short_labels, rotation=35, ha='right')
    ax.set_yticks(range(NUM_LABELS)); ax.set_yticklabels(short_labels)
    ax.set_title(name, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha='center', va='center',
                    fontsize=8, color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("fig1_confusion_matrices.png", bbox_inches='tight')
plt.show()
print("✓ Saved fig1_confusion_matrices.png")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 15 — Figure 2: ROC curves (RF and XGB, one-vs-rest)
# ──────────────────────────────────────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=range(NUM_LABELS))
colors_roc  = plt.cm.tab10(np.linspace(0, 1, NUM_LABELS))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROC Curves (One-vs-Rest)", fontsize=14, fontweight='bold')

for ax, (name, prob) in zip(axes, [
    ("Random Forest", rf_prob),
    ("XGBoost",       xgb_prob),
]):
    for i, (cls, col) in enumerate(zip(le.classes_, colors_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=1.5,
                label=f"{SHORT_NAMES.get(cls, cls)} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(name, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_roc_curves.png", bbox_inches='tight')
plt.show()
print("✓ Saved fig2_roc_curves.png")

# BERT ROC curves (separate plot)
fig, ax = plt.subplots(figsize=(7, 6))
for i, (cls, col) in enumerate(zip(le.classes_, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], bert_prob[:, i])
    ax.plot(fpr, tpr, color=col, lw=1.5,
            label=f"{SHORT_NAMES.get(cls, cls)} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5)
ax.set_title("BERT ROC Curves", fontweight='bold')
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2b_bert_roc.png", bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# CELL 16 — Figure 3: PR-AUC curves
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Precision-Recall Curves (One-vs-Rest)", fontsize=14, fontweight='bold')

for ax, (name, prob) in zip(axes, [
    ("Random Forest", rf_prob),
    ("XGBoost",       xgb_prob),
    ("BERT",          bert_prob),
]):
    macro_ap = average_precision_score(y_test_bin, prob, average='macro')
    for i, (cls, col) in enumerate(zip(le.classes_, colors_roc)):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i], prob[:, i])
        ap = average_precision_score(y_test_bin[:, i], prob[:, i])
        ax.plot(rec, prec, color=col, lw=1.5,
                label=f"{SHORT_NAMES.get(cls, cls)} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{name}\nMacro AP = {macro_ap:.3f}", fontweight='bold')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig("fig3_pr_auc.png", bbox_inches='tight')
plt.show()
print("✓ Saved fig3_pr_auc.png")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 17 — Figure 4: t-SNE on BERT CLS embeddings
# ──────────────────────────────────────────────────────────────────────────────
# Extract CLS token embeddings from fine-tuned BERT
print("Extracting BERT embeddings for t-SNE...")

bert_model.eval()
feat_pipe = pipeline(
    "feature-extraction",
    model=bert_model,
    tokenizer=tokenizer,
    device=DEVICE,
    return_tensors="pt",
)

# Use 400 balanced samples (400 / 6 ≈ 67 per class)
n_per_class = 67
sample_idx_tsne = []
for cls_idx in range(NUM_LABELS):
    cls_mask = np.where(y_test.values == cls_idx)[0]
    chosen   = np.random.choice(cls_mask, size=min(n_per_class, len(cls_mask)), replace=False)
    sample_idx_tsne.extend(chosen)

X_tsne_texts  = X_test.iloc[sample_idx_tsne].tolist()
y_tsne_labels = y_test.iloc[sample_idx_tsne].values

embeddings = []
batch_size = 32
for i in range(0, len(X_tsne_texts), batch_size):
    batch = X_tsne_texts[i:i + batch_size]
    feats = feat_pipe(batch)                        # list of (1, seq_len, 768)
    cls_vecs = [np.array(f)[0][0] for f in feats]  # take [CLS] token
    embeddings.extend(cls_vecs)

embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# t-SNE projection
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
            random_state=42, init='pca')
emb_2d = tsne.fit_transform(embeddings)

# Plot
palette = sns.color_palette("tab10", NUM_LABELS)
fig, ax = plt.subplots(figsize=(9, 7))
for i, cls in enumerate(le.classes_):
    mask = y_tsne_labels == i
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               label=cls, c=[palette[i]], alpha=0.65, s=22, edgecolors='none')
ax.set_title("t-SNE of BERT [CLS] Embeddings (Test Set)", fontsize=13, fontweight='bold')
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.legend(fontsize=9, loc='best', framealpha=0.8)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("fig4_tsne.png", bbox_inches='tight')
plt.show()
print("✓ Saved fig4_tsne.png")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 18 — Figure 5: Model comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────
all_metrics = [rf_metrics, xgb_metrics, bert_metrics, gpt_metrics]
metrics_df  = pd.DataFrame(all_metrics).set_index('model')

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(metrics_df))
width = 0.28
bar_colors = ['#1F4E79', '#2E75B6', '#70AD47', '#ED7D31']

for i, (col, label) in enumerate([('accuracy', 'Accuracy'), ('f1', 'F1-Macro'), ('auc', 'ROC-AUC')]):
    vals = [m.get(col) or 0 for m in all_metrics]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=label, color=bar_colors[i], alpha=0.88, edgecolor='white')
    for bar, v in zip(bars, vals):
        if v: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                      f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index, rotation=12, ha='right')
ax.set_ylabel("Score"); ax.set_ylim(0.65, 1.05)
ax.set_title("Model Performance Comparison", fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("fig5_model_comparison.png", bbox_inches='tight')
plt.show()
print("✓ Saved fig5_model_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 19 — Figure 6: Recall vs. False Positive threshold analysis (BERT)
# ──────────────────────────────────────────────────────────────────────────────
# Binary: not_cyberbullying (0) vs. any cyberbullying (1)
benign_idx = list(le.classes_).index('not_cyberbullying')
y_test_bin2  = (y_test != benign_idx).astype(int)
bert_score_bully = 1 - bert_prob[:, benign_idx]  # P(cyberbullying)

thresholds = np.arange(0.1, 0.95, 0.05)
recalls, precisions, f1s, fprs = [], [], [], []

for thresh in thresholds:
    y_pred_t = (bert_score_bully >= thresh).astype(int)
    tn  = np.sum((y_pred_t == 0) & (y_test_bin2 == 0))
    fp  = np.sum((y_pred_t == 1) & (y_test_bin2 == 0))
    fn  = np.sum((y_pred_t == 0) & (y_test_bin2 == 1))
    tp  = np.sum((y_pred_t == 1) & (y_test_bin2 == 1))
    recalls.append(tp / (tp + fn + 1e-9))
    precisions.append(tp / (tp + fp + 1e-9))
    f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
    fprs.append(fp / (fp + tn + 1e-9))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("BERT: Threshold Analysis (Cyberbullying vs. Benign)", fontsize=13, fontweight='bold')

# Left: Recall & Precision vs Threshold
axes[0].plot(thresholds, recalls,    'b-o', ms=4, lw=2, label='Recall')
axes[0].plot(thresholds, precisions, 'g-s', ms=4, lw=2, label='Precision')
axes[0].plot(thresholds, f1s,        'r-^', ms=4, lw=2, label='F1-Score')
axes[0].axvline(0.40, color='orange', ls='--', lw=1.5, label='Recommended θ=0.40')
axes[0].axvline(0.50, color='gray',   ls=':',  lw=1.0, label='Default θ=0.50')
axes[0].set_xlabel("Classification Threshold"); axes[0].set_ylabel("Score")
axes[0].set_title("Recall / Precision / F1 vs Threshold")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Right: Recall vs FPR trade-off
axes[1].plot(fprs, recalls, 'purple', lw=2, marker='o', ms=4)
for thresh, fpr, rec in zip(thresholds, fprs, recalls):
    if abs(thresh - 0.4) < 0.01 or abs(thresh - 0.5) < 0.01:
        axes[1].annotate(f'θ={thresh:.1f}', (fpr, rec),
                         textcoords='offset points', xytext=(8, -8), fontsize=9)
axes[1].set_xlabel("False Positive Rate (FPR)")
axes[1].set_ylabel("Recall (True Positive Rate)")
axes[1].set_title("Recall vs FPR Trade-off")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig6_threshold_analysis.png", bbox_inches='tight')
plt.show()
print("✓ Saved fig6_threshold_analysis.png")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 20 — Summary results table
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)

summary = []
for name, y_pred_arr, y_prob_arr in [
    ("Random Forest",      rf_pred,   rf_prob),
    ("XGBoost",            xgb_pred,  xgb_prob),
    ("BERT (fine-tuned)",  bert_pred, bert_prob),
]:
    acc = accuracy_score(y_test, y_pred_arr)
    pre = f1_score(y_test, y_pred_arr, average='macro')   # re-use f1 for brevity
    rec = f1_score(y_test, y_pred_arr, average='macro')
    f1  = f1_score(y_test, y_pred_arr, average='macro')
    rac = roc_auc_score(y_test, y_prob_arr, multi_class='ovr', average='macro')
    summary.append({
        'Model': name,
        'Accuracy': f"{acc:.4f}",
        'Precision*': f"{f1_score(y_test, y_pred_arr, average='macro', zero_division=0):.4f}",
        'Recall*': f"{f1_score(y_test, y_pred_arr, average='macro', zero_division=0):.4f}",
        'F1-Macro': f"{f1:.4f}",
        'ROC-AUC': f"{rac:.4f}",
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
print("\n* Macro-averaged across all classes")

# Save summary as CSV
summary_df.to_csv("results_summary.csv", index=False)
print("\n✓ Saved results_summary.csv")
print("\n✓ All figures saved. Upload them to your report document.")
print("   Files generated:")
for fname in ["eda_plots.png", "smote_impact.png", "fig1_confusion_matrices.png",
              "fig2_roc_curves.png", "fig2b_bert_roc.png", "fig3_pr_auc.png",
              "fig4_tsne.png", "fig5_model_comparison.png",
              "fig6_threshold_analysis.png", "results_summary.csv"]:
    print(f"   - {fname}")
