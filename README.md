# Quality Estimation for Machine Translation

This project focuses on estimating the quality of machine translation using a neural network model. The script trains and evaluates a regression model to predict the usability of translated text based on various features.

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- `virtualenv` package (if not installed, you can install it via `pip install virtualenv`)

### Steps

1. **Clone the Repository**
    ```sh
    git clone git@github.com:4gac/QE_for_MT.git
    cd QE_for_MT
    ```

2. **Create a Virtual Environment**
    ```sh
    python3 -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```

4. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

5. **Run the Model**
    ```sh
    python3 model.py # To train the model
    ```
### Additional Commands

- Verbose output:
  ```sh
  python3 model.py --verbose 1 
  ```
# Example output:
```sh
2024-06-19 13:15:03.129609: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
76/76 ━━━━━━━━━━━━━━━━━━━━ 0s 519us/step
76/76 ━━━━━━━━━━━━━━━━━━━━ 0s 739us/step
76/76 ━━━━━━━━━━━━━━━━━━━━ 0s 788us/step
76/76 ━━━━━━━━━━━━━━━━━━━━ 0s 792us/step
76/76 ━━━━━━━━━━━━━━━━━━━━ 0s 742us/step
K-fold Results:
Average Pearson Correlation: 0.259842402885218
Average Spearman Correlation: 0.27522501312914

76/76 ━━━━━━━━━━━━━━━━━━━━ 0s 512us/step
Results:
Pearson Correlation: 0.30266633803487825
Spearman Correlation: 0.3031066596995198
```

# Files
- model.py: The main script to create and evaluate model
- requirements.txt - The list of dependencies
- dataset_en_sk_tagget_full.tsv: Raw dataset with human annotations
- qe_dataset.sk: Dataset used to train QE model
