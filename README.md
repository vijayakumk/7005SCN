# Modelling Human Error Patterns in Air Traffic Control Using Sequence-Based Neural Networks

This repository contains the computational artefact developed for an MSc Individual Research Project investigating the classification of human-factor-related patterns in air traffic control incident narratives.

The project uses publicly available, de-identified aviation safety reports from the NASA Aviation Safety Reporting System (ASRS) and formulates the task as multi-label text classification.

## Project Overview

The study compares three text-classification approaches:

- TF-IDF with One-vs-Rest Logistic Regression
- Long Short-Term Memory (LSTM)
- Bidirectional Gated Recurrent Unit (BiGRU)

SHAP is used to provide local explainability for a selected sequence-model prediction.

After data preparation, the modelling dataset contained 8,437 narratives across nine human-factor categories.

## Human-Factor Categories

The nine target categories are:

1. Situational Awareness
2. Communication Breakdown
3. Confusion
4. Distraction
5. Workload
6. Time Pressure
7. Training / Qualification
8. Human-Machine Interface
9. Troubleshooting

## Repository Structure

```text
.
├── 7005SCN_notebook.ipynb
├── requirements.txt
├── figures/
├── tables/
└── outputs/
