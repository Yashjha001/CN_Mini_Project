# Intelligent Network Traffic Analysis Using Machine Learning

An end-to-end prototype for analyzing network traffic, classifying normal and
potentially malicious flows, and monitoring live traffic through a Streamlit
dashboard.

> **Project status:** The dataset classifier is evaluated on held-out data.
> Live capture, flow extraction, and live machine-learning predictions are
> implemented for experimentation and demonstration. Live predictions should
> not be treated as confirmed security detections until feature compatibility
> has been fully validated.

## Overview

The project combines computer networking and machine learning:

1. Learn traffic patterns from the CIC-IDS2017 dataset.
2. Clean the data and prepare a binary `Normal`/`Attack` classification task.
3. Compare Decision Tree, Random Forest, and Logistic Regression models.
4. Capture packets from a selected Windows network interface using Scapy and
   Npcap.
5. Group packets into network flows with PyFlowMeter.
6. Align live-flow features with the trained model and generate experimental
   predictions.
7. Display traffic statistics, alerts, and predictions in Streamlit.

In one sentence: this prototype learns normal-versus-attack patterns from
historical network data, captures live packets, extracts flow statistics,
produces experimental predictions, and displays the results in a monitoring
dashboard.

## Results

The selected model is a Random Forest trained on 70 retained input features.
The following results are from the held-out test set, not from live traffic:

| Metric | Result |
| --- | ---: |
| Accuracy | 99.83% |
| Attack precision | 99.54% |
| Attack recall | 99.54% |
| Attack F1-score | 99.54% |
| Correct predictions | 27,167 / 27,213 |

The test confusion matrix is:

```text
                 Predicted Normal  Predicted Attack
Actual Normal              22,216                 23
Actual Attack                  23              4,951
```

Model comparison on validation data:

| Model | Attack F1-score |
| --- | ---: |
| Decision Tree | 99.43% |
| Random Forest (selected) | 99.50% |
| Logistic Regression | 87.26% |

## Dataset preparation

The supplied CIC-IDS2017 files originally contained approximately 2.83 million
records. A 5% development sample was used:

| Stage | Records |
| --- | ---: |
| Sampled records | 141,536 |
| Duplicate rows removed | 5,475 |
| Records after deduplication | 136,061 |
| Training split | 108,848 |
| Test split | 27,213 |

The data-cleaning workflow checks missing values, infinite values, duplicate
rows, constant features, and clearly invalid negative measurements. The
initial TCP-window encoding of `-1` is preserved where it is part of the
original data convention.

## Dashboard

Run the dashboard from the repository root:

```powershell
cd IntelligentNetworkTrafficAnalysis
streamlit run app.py
```

The dashboard supports:

- Uploading a compatible network-flow CSV for classification.
- Recent packet counts and TCP/UDP activity.
- Traffic charts and recorded rule-based alerts.
- Live-flow prediction counts and model attack scores.
- Automatic refresh of live monitoring panels every three seconds.

The dashboard reads generated files from
`IntelligentNetworkTrafficAnalysis/reports/`.

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Windows live packet capture, install [Npcap](https://npcap.com/) and
enable WinPcap-compatible API support if requested by the flow-extraction
tooling. Administrative permissions may be required for packet capture.

## Workflow scripts

The numbered scripts in
`IntelligentNetworkTrafficAnalysis/notebook/` are ordinary Python scripts
(despite the directory name) and are intended to be run in order when
reproducing the pipeline:

| Script | Purpose |
| --- | --- |
| `01_dataset_inspection.py` | Inspect source dataset structure |
| `02_label_distribution.py` | Review class distribution |
| `03_prepare_binary_dataset.py` | Create the binary sample |
| `04_data_quality_check.py` | Check data quality |
| `05_clean_and_split.py` | Clean and split training/test data |
| `06_train_models.py` | Compare models and save the selected model |
| `07_live_monitor.py` | Capture packets and create heuristic alerts |
| `08_live_flow_extraction.py` | Extract flows from live capture |
| `09_process_pcap.py` | Process a PCAP file into flow features |
| `10_predict_live_flows.py` | Predict aligned live flows |
| `11_validate_live_features.py` | Compare live and training features |
| `12_audit_training_anomalies.py` | Audit unusual training values |
| `13_clean_invalid_values.py` | Create cleaned data copies |
| `14_continuous_live_ml_monitor.py` | Continuously capture and predict |

The live scripts currently contain a Windows project path
(`C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis`). Update `BASE_DIR` in
those scripts if the repository is cloned elsewhere.

Example commands:

```powershell
cd IntelligentNetworkTrafficAnalysis
python notebook\01_dataset_inspection.py
python notebook\06_train_models.py
python notebook\07_live_monitor.py
python notebook\08_live_flow_extraction.py
python notebook\10_predict_live_flows.py
```

The live monitor and dashboard can run in separate terminals. Start the
monitor first so that it can write packet and alert records while the
dashboard reads them.

## Repository layout

```text
.
├── IntelligentNetworkTrafficAnalysis/
│   ├── app.py                         # Streamlit dashboard
│   ├── data/processed/                # Prepared and split datasets
│   ├── models/                        # Saved Random Forest and feature list
│   ├── notebook/                      # Reproducible pipeline scripts
│   └── reports/                       # Metrics, captures, logs, and outputs
└── MachineLearningCVE/                # CIC-IDS2017 source CSV files
```

Important generated artifacts include:

- `models/best_traffic_model.joblib`
- `models/feature_columns.json`
- `reports/final_test_report.txt`
- `reports/live_traffic.csv`
- `reports/live_alerts.csv`
- `reports/live_flow_features_aligned.csv`
- `reports/validation_model_comparison.csv`

## Detection and safety notes

- A packet is not a flow. The classifier was trained on flow-level features,
  so individual packets must not be passed directly to the model.
- Rule-based alerts such as high packet rates, possible port scans, and TCP
  SYN bursts are configurable heuristics, not proof of an attack.
- Live-flow feature calculations still require additional equivalence
  validation against the original CIC-IDS2017 feature definitions.
- The current prototype monitors traffic visible to the selected computer
  interface; it does not monitor every device connected to a router.
- It does not automatically block traffic or provide production-grade
  incident response.
- Do not use the prototype as the sole control for security decisions.

## Data and attribution

The project uses the CIC-IDS2017 intrusion-detection dataset. Retain the
dataset's original terms, attribution, and distribution conditions when
redistributing or reusing the data. The repository includes large source and
processed CSV files through Git LFS.

## Future work

- Validate every live-flow feature against the training feature definition.
- Improve flow extraction consistency and remove environment-specific paths.
- Tune and evaluate heuristic alert thresholds on representative traffic.
- Add broader interface and router-level monitoring where appropriate.
- Establish a production evaluation protocol before treating predictions as
  security alerts.
