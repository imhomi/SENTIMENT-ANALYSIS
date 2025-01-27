# Restaurant Review Sentiment Analysis

## Project Overview
This project aims to analyze restaurant reviews and build a machine learning model to classify the sentiment of reviews as either positive (liked) or negative (not liked). The project demonstrates the process of working with textual data, applying natural language processing (NLP) techniques, and creating a sentiment classification model.

## Dataset
The dataset used for this project is **Restaurant_Reviews.tsv**, which contains customer reviews and their corresponding sentiment labels.

### Dataset Structure
| Column Name | Description                              |
|-------------|------------------------------------------|
| Review      | The textual review given by the customer |
| Liked       | Sentiment label (1 for positive, 0 for negative) |

### Sample Data
| Review                                                | Liked |
|-------------------------------------------------------|-------|
| Wow... Loved this place.                              | 1     |
| Crust is not good.                                    | 0     |
| Not tasty and the texture was just nasty.            | 0     |
| Stopped by during the late May bank holiday...        | 1     |
| The selection on the menu was great and so were...    | 1     |

## Data Analysis and Preprocessing
1. **Data Cleaning**:
   - Removed stopwords and punctuation.
   - Converted text to lowercase for uniformity.
   - Applied stemming/lemmatization to normalize words.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed the distribution of positive and negative reviews.
   - Visualized the most frequent words in positive and negative reviews using word clouds and bar plots.

## Model Development
1. **Algorithms Used**:
   - Trained and compared multiple models including:
     - Logistic Regression
     - Naive Bayes
     - Support Vector Machine (SVM)
     - K Nearest Neighbors (KNN)
     - Random Forest

2. **Best Performing Model**:
   - **SVC and Logistic Regression** achieved the highest accuracy of **77.5 %** on the test dataset.
   - Metrics used for evaluation:
     - Accuracy
     - Precision
     - Recall

3. **Model Evaluation**:
   - Provided confusion matrix and classification report to assess performance.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/inhomi/repository-name.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "skillsbuild .ipynb"
   ```

4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Dependencies
The following libraries are required to run this project:
- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- NLTK
- Jupyter Notebook

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Results
- **Model Accuracy**: [77.5%]
- **Insights**:
  - Positive reviews often use terms like "great," "loved," and "amazing."
  - Negative reviews frequently include words like "bad," "not," and "worst."

## Future Work
- Incorporate advanced NLP techniques such as word embeddings (e.g., Word2Vec, GloVe).
- Experiment with deep learning models like LSTMs or Transformers for better performance.
- Expand the dataset for improved generalization.

## Contributing
Contributions are welcome! Feel free to fork the repository, create a branch, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

