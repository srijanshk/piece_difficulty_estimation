# Piece Difficulty Estimation Challenge

### **Team Yoko Ono** : *Srijan Shakya*

This repository contains the resources for the Piece Difficulty Estimation Challenge as part of the Musical Informatics WS24 course. The project involves predicting the difficulty of musical pieces using both baseline and improved models.

---

## Repository Structure

```
📂 piece_difficulty_estimation_team_yoko_ono
├── environment.yml                  # Conda environment setup file
├── Baseline_PieceDifficulty.py      # Baseline implementation script
├── yoko_ono_PieceDifficulty.py      # Improved implementation script
├── utils.py                         # Utility functions for Report Plots
├── report.ipynb                     # Final report in Jupyter Notebook format
├── piece_difficulty_estimation.csv # Results from baseline model
├── Readme.md                        # Documentation for setup and usage
```

---

## Setup Instructions

### 1. Unzip the Submission File
```bash
unzip piece_difficulty_estimation_team_yoko_ono.zip
cd piece_difficulty_estimation_team_yoko_ono
```

### 2. Create Conda Environment
Install dependencies using the `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate miws24
```

### 3. Dataset Preparation
Ensure the dataset directory is structured as follows:
```
📂 piece_difficulty_dataset/
├── scores_difficulty_estimation/
│   ├── *.musicxml                       # MusicXML files for training and testing
├── difficulty_classification_test_no_labels.csv
├── difficulty_classification_training.csv
```

The dataset folder is not included in this repository due to submission requirements. Place the dataset in a folder named `piece_difficulty_dataset` in the root directory before running the scripts.

---

## How to Run

### Baseline and Improved Models
Run the provided scripts to perform difficulty estimation:

#### Baseline Model
```bash
python Baseline_PieceDifficulty.py -i path/to/data/ -o baseline_results_difficulty_estimation.csv
```

#### Improved Model
```bash
python yoko_ono_PieceDifficulty.py -i path/to/data/ -o improved_results_difficulty_estimation.csv
```

### Outputs
1. **baseline_results_difficulty_estimation.csv**: Predictions and metrics from the baseline model.
2. **improved_results_difficulty_estimation.csv**: Predictions and metrics from the improved model.

### Visualizations and Report
- **Confusion Matrices**: Visualize baseline vs. improved model performance using plotting functions in the scripts.
- **Final Report**: Open `report.ipynb` for detailed analysis:
  ```bash
  jupyter notebook report.ipynb
  ```

---

## Key Files

### `utils.py`
- Contains utility functions for report plots.

### `Baseline_PieceDifficulty.py`
- Implements:
  - Baseline difficulty estimation using simple features (e.g., note density).
  - Decision Tree Classifier for prediction.
- Outputs accuracy and F1-score.

### `yoko_ono_PieceDifficulty.py`
- Implements:
  - Improved difficulty estimation using advanced features.
  - Random Forest Classifier with feature importance analysis.
- Outputs enhanced metrics compared to the baseline.

---

## Notes and Limitations
- **MusicXML Dataset**: The dataset must be in MusicXML format and properly structured in the `piece_difficulty_dataset` folder.
- **Training and Testing Split**: Ensure proper data splits to avoid leakage during evaluation.

---

## Contact
For queries, reach out to **Team Yoko Ono**.
