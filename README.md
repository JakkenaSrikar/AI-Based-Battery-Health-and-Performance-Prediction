# AI-Based Battery Health and Performance Prediction for EVs

This repository contains an AI-driven approach utilizing a **Multilayer Perceptron (MLP)** model to estimate key performance indicators of an electric vehicle (EV) battery, such as **State of Charge (SoC), State of Health (SoH), Operational Duration, and Speed**.

![EV Charging](https://static.vecteezy.com/system/resources/previews/025/733/581/original/electric-car-at-charging-station-abstract-electric-power-charger-ev-clean-energy-alternative-energy-electric-charger-concept-electronic-vehicle-power-dock-illustration-vector.jpg)

## Table of Contents
- [Introduction](#introduction)
- [Model Structure](#model-structure)
- [Dataset Details](#dataset-details)
- [How to Use](#how-to-use)
- [Future Enhancements](#future-enhancements)
- [Contribution Guidelines](#contribution-guidelines)

## Introduction

This project applies a **Multilayer Perceptron (MLP)** architecture to analyze EV battery parameters and predict:
- **State of Charge (SoC)** – Percentage of charge left in the battery.
- **State of Health (SoH)** – Overall battery condition as a percentage.
- **Operational Duration** – Estimated runtime in hours.
- **Speed** – Expected vehicle speed based on battery performance.

## Model Structure

The model follows a **multi-input, multi-output MLP** framework, where each branch is dedicated to predicting a specific battery parameter. The structure comprises:
- Fully connected layers utilizing **LeakyReLU activation** for non-linearity.
- Distinct output layers for each predicted metric.

## Dataset Details

The dataset is designed around key battery characteristics:
- **Open-Circuit Voltage (Voc)**: Measured voltage without load.
- **Remaining Energy (kWh)**: Available energy in the battery.
- **Battery Current (A)**: Current draw from the battery.
- **Operation Duration (hrs)**: Estimated working hours.
- **Battery Range at Various Levels**: Battery performance at 150km, 145km, and 140km benchmarks.

### Key Constants for Battery Estimation
To ensure accurate feature representation, the dataset includes:
- **Maximum Battery Voltage**: 58.8V
- **Minimum Battery Voltage**: 38.5V

Before running the model, ensure the dataset aligns with these features.

## How to Use

### 1. Prepare the Input Data
- Store the data in a CSV file, e.g., [`dataset.csv`](dataset.csv).
- The file should include all necessary battery parameters.
- Clean the data by handling missing values before model execution.

### 2. Running the Model
- Execute the script in a Jupyter Notebook or Python environment.
- The script includes:
  - Data loading and preprocessing steps.
  - MLP model definition and training.
  - Performance evaluation on test data.

## Future Enhancements

### Adaptive Self-Training Mechanism
To enhance accuracy, a self-improving training mechanism can be introduced:
1. **Prediction on New Data**: Use the trained model to infer SoC, SoH, duration, and speed on fresh data.
2. **Confidence-Based Selection**: Filter predictions exceeding a confidence threshold (e.g., 90%).
3. **Incremental Training**: Incorporate high-confidence samples into the training dataset for model updates.

## Contribution Guidelines

We welcome contributions! If you have suggestions, feature enhancements, or bug fixes, feel free to submit a pull request or open an issue.

---

This project aims to enhance battery health monitoring and optimize EV performance through AI-driven analytics.

