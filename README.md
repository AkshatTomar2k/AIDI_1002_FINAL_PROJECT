𝐄𝐱𝐩𝐥𝐨𝐫𝐢𝐧𝐠 𝐁𝐄𝐑𝐓 𝐕𝐚𝐫𝐢𝐚𝐧𝐭𝐬 𝐚𝐧𝐝 𝐇𝐲𝐩𝐞𝐫𝐩𝐚𝐫𝐚𝐦𝐞𝐭𝐞𝐫 𝐓𝐮𝐧𝐢𝐧𝐠 𝐟𝐨𝐫 𝐒𝐞𝐧𝐭𝐞𝐧𝐜𝐞 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐜𝐚𝐭𝐢𝐨𝐧 𝐨𝐧 𝐭𝐡𝐞 𝐌𝐑𝐏𝐂 𝐃𝐚𝐭𝐚𝐬𝐞𝐭

𝐎𝐯𝐞𝐫𝐯𝐢𝐞𝐰

This repository contains the code and resources for exploring BERT variants and hyperparameter tuning in the context of sentence classification using the Microsoft Research Paraphrase Corpus (MRPC) dataset. The primary goal is to analyze how different hyperparameters and model variants affect performance and provide insights into best practices for model selection and tuning in NLP tasks.

**Repository_Structure**

data/: Contains the dataset and data preprocessing scripts.
notebooks/: Jupyter notebook with detailed experimental setup, results, and analysis.
scripts/: Python scripts for model training, evaluation, and hyperparameter tuning.
results/: Folder containing output files and result summaries.
README.md: This file.
requirements.txt: List of required Python packages.

**Installation**

To run the code, you need to have Python 3.7 or later installed. Follow the steps below to set up your environment:

Clone the Repository
bash
Copy code
git clone 
cd bert-hyperparameter-tuning

Create and Activate a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install Dependencies
bash
Copy code
pip install -r requirements.txt

**Dataset**
The dataset used in this project is the Microsoft Research Paraphrase Corpus (MRPC). The dataset is divided into training and validation sets as follows:

Training Set: 3,500 sentence pairs
Validation Set: 1,700 sentence pairs
The dataset is available in the data/ directory, and preprocessing scripts are included to prepare the data for model training.

**Usage**
1. Data Preparation
To prepare the dataset for training, run the following script:

bash
Copy code
python scripts/preprocess_data.py
This script will tokenize and format the MRPC dataset for input into the BERT model.

2. Training the Model
To train the BERT model with different hyperparameter settings, use the train_model.py script. You can specify hyperparameters such as learning rate, batch size, and number of epochs through command-line arguments.

Example:

bash
Copy code
python scripts/train_model.py --learning_rate 2e-5 --batch_size 16 --num_epochs 3
3. Evaluation
To evaluate the trained model, use the following command:

bash
Copy code
python scripts/evaluate_model.py
This script will generate performance metrics including accuracy, precision, recall, and F1 score.

4. Jupyter Notebook
For an interactive exploration of the experiments and results, open the Jupyter notebook located in the notebooks/ directory. You can run the notebook with:

bash
Copy code
jupyter notebook notebooks/bert_experiments.ipynb
Hyperparameter Tuning
The following hyperparameters were explored in this project:

Learning Rate: Tested values of 1e-5, 2e-5, and 3e-5.
Batch Size: Tested values of 8, 16, and 32.
Number of Epochs: Tested values of 2, 3, and 4.
The results and analysis of these hyperparameter settings are detailed in the Jupyter notebook and summarized in the results/ directory.

**Results**
The results of the experiments, including performance metrics for different hyperparameter configurations and model variants, are available in the results/ directory. The comparative analysis includes:

Accuracy of BERT-Base vs. BERT-Large

Impact of Learning Rate on model performance

Effects of Batch Size and Number of Epochs
