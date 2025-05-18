Link to the solution: https://github.com/lathamk4/Capstone_MT_20_1/blob/main/Capstone.ipynb

Dataset & reference: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

# Goal:
The goal of this project is to build a machine learning model that can accurately predict the medical specialty based on the content of medical transcriptions. 
This will help automate classification, streamline healthcare workflows, and support faster and more accurate specialist referrals.
Few Classifier Models under consideration:
1)Logistic Regression 
2)Multinomial Naive Bayes
3)Support Vector Machine (SVM)
4)Random Forest 
5)Transformer-based models (e.g., BERT)

# Business Objective: 
The objective is to automate classification of medical specialties from transcriptions to streamline clinical workflows. This enhances care coordination by routing information to the correct specialists. 
It also supports efficient resource allocation and operational planning. Ultimately, it improves patient outcomes and reduces administrative burden.

# Exploratory Data Analysis: 
As part of the data analysis following operations were performed 

#1) Data Cleaning: 
For this project, we are considering only the transcription column and medical speciatly as the traget label columns. Hence dropping all other columns. 
There are 33 null values in the Specialty column which is filled with 'unknown' . 

#2)Text Preprocessing:
Lowercasing , Removing Punctuation and Stop words were used. Tokenization and Lemmatization has to be implemented but Punkt is not downloding properly. Working on fixing the errors. 

#3) Data Visualization: 
a) Distribution of medical specialty: 
This plot helps find any imbalance in the dataset and based on the observation we can use techniques like oversampling or undersampling to overcome the bias or variance
![1](https://github.com/user-attachments/assets/26947e16-48e1-4606-bcc8-9dd9e38bb9ed)

b) Trasncription length by Medical specialty: 
This plot helps identify which specialties produce longer or more detailed notes, guiding preprocessing and model input length. It also reveals content variability, useful for customizing NLP models per specialty.
![2](https://github.com/user-attachments/assets/21427796-18b2-4c34-9efe-c3d078654c7b)
c) TOP TF-IDF terms by medical specialty plot :
![3](https://github.com/user-attachments/assets/f47ee9ea-5f73-433d-84c9-5220f262a57f)

d) Top 10 most common words per Medical specialty plot:
![4](https://github.com/user-attachments/assets/c3594832-0fb4-4b4a-b016-8e92f55bb828)

Both the plot helps identifying unique words specific to specialty thereby enhacing the feature extraction and specialty specific text classification
e) word cloud by transcriptions : 
![5](https://github.com/user-attachments/assets/9362656c-d89c-49e8-ac90-d35e78af249c)

This plot helps visualize most frequent terms among all the specialities and aids in the quick exploratory analysis 
f) Top 10 Most frequent word specific to specialty 
![6](https://github.com/user-attachments/assets/3153c211-6db4-41fa-97e8-c99e11c7cb65)

This plot helps with identifying the specialty-specific language pattern which in turn improve model performance.

#4) Feature Engineering : 
Identifying and extract terms specific to the medical specialty using the clinical BERT 
Identify the NER to extract entities like patient age and sex and remove from transcriptions.

#5 Initial Modeling:
The Multinomial Naive Bayes classifier model is used for training and Accuracy score was just 37%. 

# Findings:
Extremely Imbalanced Dataset:
Several medical specialties have only 1–2 examples, making it difficult for the model to learn any meaningful patterns from those categories.

Underfitting:
The current Multinomial Naive Bayes model appears too simple for the complexity of this classification task. It ends up focusing on a few majority classes while ignoring the rest.

Zero F1-Scores for Most Classes:
As shown in the classification report, the model fails to predict the majority of classes, resulting in zero precision, recall, and F1 scores for many categories.

![7](https://github.com/user-attachments/assets/487aa8cd-4aab-4589-afe9-2477282a7433)


# Next Steps / Ideas for Improvement:
1. Class Grouping or Rebalancing
Consider merging infrequent medical specialties into an “Other” or “Miscellaneous” category.

Alternatively, drop very rare classes and focus on the most common 10–15 specialties to simplify the classification task initially.

2. Resampling Techniques
Use techniques like oversampling (e.g., RandomOverSampler) or undersampling to balance the class distribution in the training data.

3. Stronger Classification Models
Explore more expressive models that can better handle text features and class imbalances:
 
    Logistic Regression – baseline linear classifier that often performs well on text
    
    Support Vector Machine (SVM) – good for high-dimensional sparse data (e.g., TF-IDF vectors)
    
    Random Forest – may handle class imbalance better, though slower on high-dimensional data
    
    Transformer-based Models – like BERT, which can capture complex language patterns and context better than traditional models

4.Feature Engineering with other Vectorization Techniques like CountVectorizer, ClinicalBERT transformers.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Capstone 24.1 

# Problems : 
Few challenges associated with finding the right machine learning solution are listed below:

1. Severe Class Imbalance
Some medical specialties (e.g., "Surgery") have hundreds of transcriptions, while others (e.g., "Allergy / Immunology") have fewer than 10.

2. Noisy and Overlapping Content
Transcriptions often contain overlapping content across specialties (e.g., general terms or shared procedures).This reduces model discriminative power and increases confusion between classes.

3. Lack of Standardization
Text formatting varies widely: inconsistent punctuation, abbreviations, and non-standard medical terms. Requires extensive preprocessing (tokenization, lemmatization, stop-word removal).

4. Unverified Clinical Accuracy
Dataset is not clinically validated; content may not reflect real-world diagnostic or transcription standards.Limits applicability in real healthcare ML systems due to potential misinformation.

# Steps taken to mitigate the problems : 

1. Unique words are identified for each row and models were trained using the unique words
2. Resampling Techniques: Applied methods like Random Sampler to balance the class distribution.
3. Data Augmentation: Generate synthetic samples (SMOTE) was used to increase the diversity of training data.
4. Advanced Models: used transformer-based models, (BioBERT)  which are well-suited for medical text classification tasks.

# Results: 
As mentioned in the Capstone 20.1 , different sampler techniques were used with different models . The accuracy comparison is as follows 

![24_1_1](https://github.com/user-attachments/assets/004609ea-c198-4d82-8c26-ec96ee64e5e9)

**Below are the evaluation metrics using Bio Bert with weighted loss**

Training Loss: 2.2628
Validation Accuracy: 32.15%
Validation F1 Score (Macro): 25.37%

# Important findings

1) Random sampling maintains the original data distribution, which seems to perform best with classical ML models.
2) SMOTE is likely generating low-quality synthetic samples, since its text data . May be numerical data or embeddings (BERT) with SMOTE might work better, even then its risky.
3) BERT captures contextual semantics, which benefits deeper models or those with dense input support (i.e) Logistic Regression and Random Forest. 
   Naive Bayes is incompatible due to assumption of discrete count-based features and SVM underperforms here possibly due to overfitting or improper kernel choice.
4) The BioBERT model with a weighted loss function demonstrated limited performance, with a validation accuracy of 32.15% and F1 score of 25.37%. The results highlight the challenges of applying deep learning to imbalanced and noisy clinical text, and suggest that further improvements require a combination of data cleaning, architectural tuning, and task-specific adaptation strategies.
   
Overall, Random Sampling worked better than SMOTE and BERT Embeddings, although results of BERT embeddings were promising.Traditional models like LR, SVM, RF perform well on unbalanced data but might get biased toward dominant classes. Hence **SVM with Random sampler seems to be a good candidate to use with further fine tuning (Cross Validation and Hyperparameter Optimization techniques).**

# suggestions for improvement : 

1) Preprocessing:	Strip headers, anonymization tags, and extraneous text; limit inputs to most informative sections (e.g., first 256 tokens).
2) Feature Engineering: Incorporate domain-specific features to enhance model understanding.
3) Label Structure:	Review and consider merging similar specialty labels based on semantic overlap.
4) Data Augmentation: Generate synthetic samples to increase the diversity of training data.
5) Expert Review: Collaborate with medical professionals to validate and refine the dataset, ensuring its clinical relevance and accuracy.
6) Error Analysis: Generate and analyze a confusion matrix to identify class-specific misclassifications.
7) For Clinical BERT , Increase training epochs to 5–10; apply learning rate warm-up and decay; consider gradient clipping.

# Additional Steps performed: 
1) Model Saving: Store associated preprocessing objects (e.g., tokenizers, encoders) to ensure consistency during inference. Helps to avoid retraining , Imporves efficiency and makes it deployment ready
2) Inference Pipeline: Set up a pipeline that takes new, unseen medical transcriptions and processes them through the same preprocessing steps.Load the saved model and generate predictions for the appropriate medical specialty.
3) Package the model and inference pipeline into a service. FASTAPI was used . This step ensures low-latency, scalable responses for real-world usage. 

