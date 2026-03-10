# Tennessee Eastman Process Anomaly Detection

This project presents a robust, end-to-end machine learning pipeline for detecting chemical process anomalies within the **Tennessee Eastman Process (TEP)** dataset using:

- **Dynamic Fault Detection** to identify continuous process disturbances across multiple sensor variables.
- **Robust Preprocessing** pipelines to handle missing values, scale sensor readings, and manage time-series sequencing.
- **Anomaly Scoring** via  to reconstruct expected states and flag high-error deviations.
- **Experiment Tracking**, ensuring reproducibility and comprehensive model evaluation across different fault types.

---

## Pipeline Overview

```mermaid
flowchart TD
    A[Raw TEP Data] --> B[Data Preprocessing & Scaling]
    B --> C[Feature Engineering / Sequencing]
    C --> D[Anomaly Detection Model]
    D --> E[Reconstruction Error / Scoring]
    E --> F[Threshold Tuning]
    F --> G[Final Fault Classification]
