# Modelling Human Error Patterns in Air Traffic Control Using Sequence-Based Neural Networks

## MSc Individual Research Project

This repository contains the final computational artefact developed for the MSc dissertation:

**“Modelling Human Error Patterns in Air Traffic Control Using Sequence-Based Neural Networks.”**

The project investigates whether sequence-based neural networks can identify human-factor-related patterns from air traffic control incident narratives and whether explainability methods can provide insight into selected model predictions.

Publicly available, de-identified aviation safety reports from the **NASA Aviation Safety Reporting System (ASRS)** were used. The task was formulated as **multi-label text classification**, where a single incident narrative may contain more than one human-factor category.

The implemented approaches are:

- TF-IDF with One-vs-Rest Logistic Regression
- Long Short-Term Memory (LSTM)
- Bidirectional Gated Recurrent Unit (BiGRU)
- SHAP-based local explainability

---

## Research Aim

The project aimed to investigate the use of sequence-based neural networks for modelling human-factor-related patterns in air traffic control incident narratives and to examine how explainability techniques can support interpretation of model predictions.

---

## Research Questions

1. How effectively can sequence-based neural networks classify human-factor-related patterns within air traffic control incident narratives?

2. How do LSTM and BiGRU architectures compare in their ability to model these patterns?

3. How can explainability techniques be used to interpret the human-factor predictions produced by the selected sequence model?

---

## Data Source

The project uses reports retrieved from the **NASA Aviation Safety Reporting System Database Online**:

https://asrs.arc.nasa.gov/search/database.html

The study covers reports from **2009 to 2025**.

Three ASRS exports were used:

| Period | Reports |
|---|---:|
| 2009–2013 | 3,671 |
| 2014–2017 | 3,045 |
| 2018–2025 | 3,089 |
| **Initial total** | **9,805** |

Following validation, one out-of-period record was removed, leaving **9,804 records** before modelling selection.

Reports containing a usable narrative and at least one selected human-factor category were retained for modelling, resulting in:

**8,437 reports**

### Data availability

The original ASRS CSV exports and processed narrative datasets are **not included in this repository**.

The data can be obtained from the NASA ASRS Database Online using the source link above.

This repository contains the analysis notebook and non-sensitive analytical outputs required to understand the implementation and results.

---

## Human-Factor Categories

Nine human-factor categories were retained for modelling:

1. Situational Awareness
2. Communication Breakdown
3. Confusion
4. Distraction
5. Workload
6. Time Pressure
7. Training / Qualification
8. Human-Machine Interface
9. Troubleshooting

The prediction task is **multi-label**, meaning that an individual report can contain multiple human-factor categories simultaneously.

---

## Dataset Preparation

The original ASRS exports contain a two-level header structure. The implementation reconstructs these headers into distinct Python-compatible variable names while preserving the relationship between sections such as `Person 1` and `Report 1`.

The modelling experiment uses:

- **Input:** Report 1 Narrative
- **Targets:** selected Person 1 Human Factors

The final target matrix has the shape:

```text
(8437, 9)
```

---

## Experimental Split

The data were divided using **iterative multi-label stratification** to preserve label prevalence across the experimental subsets.

| Subset | Reports |
|---|---:|
| Training | 5,929 |
| Validation | 1,254 |
| Test | 1,254 |
| **Total** | **8,437** |

Approximate split:

- Training: 70%
- Validation: 15%
- Test: 15%

The validation set was used for model-development decisions and label-specific threshold selection.

The test set was reserved for final performance evaluation.

A fixed random seed of:

```text
2026
```

was used during the data-splitting procedure.

---

## Classification Approaches

### 1. TF-IDF + Logistic Regression

A conventional text-classification baseline was implemented using:

- TF-IDF representation
- maximum 30,000 features
- unigrams and bigrams
- minimum document frequency of 2
- maximum document frequency of 0.98
- One-vs-Rest classification
- balanced Logistic Regression

The TF-IDF vectoriser was fitted using the **training set only** before transforming the validation and test data.

---

### 2. Long Short-Term Memory

The LSTM architecture consists of:

- learned embedding layer
- embedding dimension: 64
- SpatialDropout1D: 0.30
- 64-unit LSTM layer
- L2 regularisation
- 64-unit dense layer with ReLU activation
- dropout: 0.40
- nine sigmoid output units

Final LSTM parameter count:

```text
805,769
```

---

### 3. Bidirectional Gated Recurrent Unit

The BiGRU architecture uses:

- learned embedding layer
- embedding dimension: 64
- SpatialDropout1D: 0.30
- bidirectional 64-unit GRU
- L2 regularisation
- 64-unit dense layer with ReLU activation
- dropout: 0.40
- nine sigmoid output units

Final BiGRU parameter count:

```text
826,761
```

---

## Text Representation for Sequence Models

The sequence-model vocabulary was learned using **training narratives only**.

Configuration:

| Parameter | Value |
|---|---:|
| Maximum vocabulary size | 12,000 |
| Sequence length | 700 tokens |
| Embedding dimension | 64 |

The sequence length was selected after examining the training-set narrative-length distribution. The 95th percentile was approximately **673 words**, which was rounded to a practical sequence length of **700 tokens**.

---

## Class Imbalance

The nine human-factor categories occur at substantially different frequencies.

To reduce the influence of this imbalance, the sequence models use a **class-sensitive multi-label loss function**, applying separate positive and negative weights for each label.

The Logistic Regression baseline uses:

```python
class_weight="balanced"
```

---

## Sequence-Model Training

The main sequence-model training configuration was:

| Parameter | Value |
|---|---:|
| Optimiser | Adam |
| Initial learning rate | 0.0005 |
| Batch size | 64 |
| Maximum epochs | 15 |
| Training objective | Balanced multi-label binary cross-entropy |
| Monitoring metric | Validation PR-AUC |

Training also used model checkpointing and early-stopping controls.

---

## Label-Specific Decision Thresholds

A fixed probability threshold of 0.50 was not assumed for every target.

Instead, a separate classification threshold was selected for each human-factor category using the **validation set**.

Candidate thresholds were evaluated between:

```text
0.10 and 0.80
```

in increments of:

```text
0.02
```

The threshold producing the highest validation F1-score for each label was retained and subsequently applied to test predictions.

---

## Evaluation Metrics

Because the task is multi-label and the target categories are imbalanced, model performance was assessed using several complementary measures:

- Subset Accuracy
- Hamming Loss
- Micro Precision
- Micro Recall
- Micro F1-score
- Macro F1-score
- Weighted F1-score
- Macro ROC-AUC
- Macro Average Precision

Particular attention was given to **Macro F1**, because it gives equal importance to each target category rather than allowing the most frequent labels to dominate the result.

---

## Final Test Results

| Model | Subset Accuracy | Hamming Loss | Micro Precision | Micro Recall | Micro F1 | Macro F1 | Weighted F1 | Macro ROC-AUC | Macro Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.0622 | 0.2727 | 0.5107 | 0.8104 | 0.6265 | **0.5200** | 0.6379 | **0.7552** | **0.5204** |
| LSTM | 0.0167 | 0.4049 | 0.3982 | 0.8500 | 0.5424 | 0.4328 | 0.5772 | 0.6041 | 0.3602 |
| BiGRU | 0.0191 | 0.4215 | 0.3895 | **0.8688** | 0.5378 | 0.4380 | 0.5793 | 0.6181 | 0.3608 |

### Main finding

The **TF-IDF + Logistic Regression baseline achieved the strongest overall test performance**, including the highest:

- Micro F1
- Macro F1
- Macro ROC-AUC
- Macro Average Precision

The sequence models achieved higher recall but lower precision, indicating that they detected a large proportion of positive labels while also producing more false-positive classifications.

---

## LSTM vs BiGRU

Model selection between the two sequence architectures was performed using **validation Macro F1**, rather than test-set performance.

| Model | Validation Macro F1 |
|---|---:|
| LSTM | 0.4469 |
| **BiGRU** | **0.4527** |

The BiGRU was therefore selected as the sequence model for detailed error analysis and explainability.

This selection does **not** mean that the BiGRU outperformed the TF-IDF baseline overall. It only indicates that it achieved the stronger validation Macro F1 of the two recurrent architectures.

---

## Training Behaviour

For the LSTM:

- peak validation PR-AUC: **0.3588**
- peak occurred at epoch: **7**
- training PR-AUC continued to approximately **0.6213 by epoch 11**

For the BiGRU:

- peak validation PR-AUC: **0.3650**
- peak occurred at epoch: **5**

The divergence between training and validation performance indicates limited generalisation and increasing overfitting beyond the strongest validation epochs.

---

## Error Analysis

The selected sequence model was examined at report level by counting false-positive and false-negative labels.

The two test reports with the highest total number of label errors were:

| ACN | False Positives | False Negatives | Total Errors |
|---|---:|---:|---:|
| 1221260 | 8 | 1 | 9 |
| 2207767 | 8 | 1 | 9 |

The error-analysis stage was used to identify difficult reports and better understand the limitations of aggregate classification metrics.

Full narrative-level error-analysis exports are not included in the public repository.

---

## Explainability

SHAP was applied to the selected **BiGRU** model to examine the contribution of narrative terms to an individual prediction.

A correctly classified test report was selected for the:

**Training / Qualification**

category.

Example:

| Item | Value |
|---|---|
| ASRS ACN | 2312357 |
| Predicted class | Training / Qualification |
| Predicted probability | 0.990 |
| Decision threshold | 0.76 |
| Prediction result | Correct positive |

Terms associated with training activity contributed strongly to the selected prediction.

The explanation is interpreted as a **local explanation of model behaviour**. SHAP values indicate how features influenced the model prediction and should not be interpreted as evidence that those terms caused the underlying aviation incident.

---

## Repository Structure

```text
atc-human-factor-sequence-models/
│
├── README.md
├── 7005SCN_notebook.ipynb
├── requirements.txt
│
├── figures/
│   ├── reports_by_year.png
│   ├── selected_variable_missingness.png
│   ├── human_factor_frequency.png
│   ├── narrative_length_distribution.png
│   ├── labels_per_report.png
│   ├── human_factor_cooccurrence.png
│   ├── lstm_loss_curve.png
│   ├── bigru_loss_curve.png
│   ├── lstm_pr_auc_curve.png
│   ├── bigru_pr_auc_curve.png
│   ├── lstm_bigru_per_label_f1.png
│   ├── lstm_confusion_matrices.png
│   ├── bigru_confusion_matrices.png
│   └── shap_local_word_contributions.png
│
├── tables/
│   └── selected analytical tables generated by the notebook
│
└── outputs/
    └── experiment settings and supporting outputs
```

The exact contents of the `figures`, `tables`, and `outputs` directories may depend on which generated supporting outputs are retained in the final repository.

---

## Notebook Workflow

The final notebook follows the approximate workflow below:

```text
NASA ASRS CSV exports
        │
        ▼
Source validation
        │
        ▼
Two-level header reconstruction
        │
        ▼
Data cleaning and validation
        │
        ▼
Exploratory data analysis
        │
        ▼
Human-factor category selection
        │
        ▼
Multi-label target construction
        │
        ▼
Train / validation / test split
        │
        ├───────────────────────────────┐
        ▼                               ▼
TF-IDF + Logistic Regression      TextVectorization
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                            LSTM                BiGRU
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                              Multi-label evaluation
                                        │
                                        ▼
                              Sequence-model selection
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                         Error analysis       SHAP analysis
```

---

## Running the Project

### Recommended environment

The project was developed and executed using **Google Colab**.

The final notebook is:

```text
7005SCN_notebook.ipynb
```

### 1. Obtain the NASA ASRS data

Retrieve the required records from:

https://asrs.arc.nasa.gov/search/database.html

The notebook expects three source files:

```text
ASRS_DBOnline_2009-2013.csv
ASRS_DBOnline_2014-2017.csv
ASRS_DBOnline_2018-2025.csv
```

### 2. Create the project directory

The notebook is configured to use the following Google Drive location:

```text
/content/drive/MyDrive/ATC_Dissertation/
```

Place the original CSV files inside:

```text
/content/drive/MyDrive/ATC_Dissertation/data/raw/
```

The expected structure is:

```text
ATC_Dissertation/
│
├── data/
│   └── raw/
│       ├── ASRS_DBOnline_2009-2013.csv
│       ├── ASRS_DBOnline_2014-2017.csv
│       └── ASRS_DBOnline_2018-2025.csv
│
├── figures/
├── tables/
├── models/
├── outputs/
└── data/
    └── processed/
```

The required output directories are created automatically by the notebook if they do not already exist.

### 3. Open the notebook in Google Colab

Upload or open:

```text
7005SCN_notebook.ipynb
```

### 4. Connect Google Drive

Run the initial Google Drive mounting cell and authorise access.

### 5. Run the notebook sequentially

Execute the cells from beginning to end.

The notebook performs:

1. source-file validation
2. ASRS header reconstruction
3. data cleaning
4. exploratory analysis
5. human-factor target preparation
6. iterative multi-label splitting
7. TF-IDF baseline modelling
8. sequence text vectorisation
9. LSTM training
10. BiGRU training
11. test-set evaluation
12. model comparison
13. error analysis
14. SHAP explainability
15. generation of analytical outputs

---

## Dependencies

The principal Python packages used are:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
iterative-stratification
joblib
shap
```

They are listed in:

```text
requirements.txt
```

The notebook also contains installation commands for packages that may not be available by default in Google Colab.

---

## Generated Outputs

The notebook generates several forms of analytical evidence.

### Exploratory analysis

Examples include:

- reports by year
- missing information in selected variables
- human-factor category frequency
- narrative-length distribution
- labels per report
- human-factor co-occurrence

### Model evaluation

Examples include:

- overall model-performance comparison
- per-label precision, recall and F1
- LSTM and BiGRU training curves
- per-label F1 comparison
- multi-label confusion matrices
- validation-selected decision thresholds
- report-level error analysis

### Explainability

The notebook also produces a SHAP-based word-level explanation for a selected BiGRU prediction.

---

## Reproducibility Notes

The repository provides the final analysis notebook and supporting outputs used for the dissertation.

Several points should be considered when reproducing the study:

- the same NASA ASRS records and filtering criteria are required;
- the ASRS source exports must retain the expected two-level header format;
- the data-splitting procedure uses a fixed random seed;
- validation data are used for label-specific threshold selection;
- test data are reserved for final reporting;
- neural-network training includes stochastic optimisation, so an independent rerun may not reproduce every metric to the final decimal place;
- results reported in the dissertation correspond to the completed notebook run contained in this repository.

---

## Ethical Considerations

NASA ASRS reports are publicly available and are de-identified before release.

This study:

- uses only publicly released ASRS data;
- does not attempt to identify reporters;
- does not attempt to reconstruct information removed during de-identification;
- treats human-factor categories as report annotations rather than definitive assessments of individual competence;
- uses model predictions for retrospective research analysis only.

---

## Intended Use

The developed models are **research artefacts**, not operational aviation-safety systems.

They are not intended to:

- make autonomous safety decisions;
- evaluate individual air traffic controllers;
- determine responsibility or blame;
- provide causal explanations of aviation incidents;
- replace professional human-factors or aviation-safety analysis.

Any future operational application would require substantially broader validation, expert review and appropriate governance.

---

## Key Limitations

Important limitations of the study include:

- ASRS reporting is voluntary;
- narrative content may vary substantially in detail and quality;
- the human-factor annotations are not independent psychological ground truth;
- the nine selected target categories remain imbalanced;
- multiple human-factor categories may overlap within the same report;
- the predictive experiment uses narrative text rather than combining text with structured operational variables;
- the recurrent models showed evidence of limited generalisation;
- neural-network experiments were not repeated across multiple independent training seeds;
- evaluation was conducted using data from the same underlying ASRS source.

The results should therefore be interpreted as evidence of classification performance within the study dataset rather than as evidence of real-time operational performance.

---

## Project Status

**Final dissertation artefact**

The notebook in this repository represents the final implementation used to produce the reported dissertation results.

---

## Author

**YOUR_NAME**

MSc Artificial Intelligence and Human Factors  
Coventry University

GitHub: `https://github.com/YOUR_GITHUB_USERNAME`

---

## Academic Context

This repository accompanies the MSc Individual Research Project submitted as part of the **MSc Artificial Intelligence and Human Factors** programme at Coventry University.

The repository is provided to support transparency, inspection of the computational artefact and reproducibility of the research workflow.

---

## Data Attribution

Aviation safety report data were obtained from:

**NASA Aviation Safety Reporting System (ASRS)**  
https://asrs.arc.nasa.gov/

NASA ASRS retains responsibility for the original source database. The analysis, preprocessing, modelling and interpretation in this repository form part of the author's academic research project.

---

## Licence

No open-source licence is currently attached to this repository.

The original NASA ASRS data are not redistributed here.

Unless otherwise stated, the original analysis code and project materials remain the work of the repository author.
