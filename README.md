# AI Driven Political Bias Detection and Analysis - Hugging Face

## Abstract
This project investigates the effectiveness of AI-driven tools in detecting and categorizing political bias in news and social media content. By leveraging models such as GPT-4, VADER, and PoliticalBiasBERT, we analyze their ability to identify linguistic patterns and emotional tones that signal left-leaning, center, or right-leaning biases. Using a curated dataset of political articles, social media posts, and candidate statements, we evaluate the accuracy and limitations of these tools in politically sensitive contexts. Our findings reveal that while these tools excel at detecting overt biases, they struggle with nuanced or neutral content. We recommend combining multiple models, refining training data, and incorporating human oversight to enhance the reliability and interpretability of AI in political bias detection. This study provides valuable insights into the potential and challenges of using AI for promoting fairness and transparency in political discourse.

## Overview
This script uses the Hugging Face Transformers library and a fine-tuned BERT model (bucketresearch/politicalBiasBERT) to classify political bias in text files as Left, Center, or Right.

### Features
* Large Text Handling: Automatically splits large texts into manageable 512-token chunks (BERT's maximum input size).
* Probabilistic Averaging: Averages class probabilities across chunks for an overall document-level prediction.
* Reusable Functionality: Includes a modular classify_political_bias function for flexible text classification.

### How It Works

1. Preprocessing:
* The script reads a text file and tokenizes the content.
* Large texts are split into 512-token chunks.

2. Classification:
* Each chunk is passed through the BERT model to generate class probabilities for Left, Center, and Right.
* The script uses softmax normalization to convert model logits into probabilities.

3. Aggregation:
* Probabilities from all chunks are averaged to produce an overall classification for the document.
* The class with the highest average probability is selected as the final prediction.

4. Output:
* Chunk-level predictions: Displays probabilities and the predicted class for each chunk.
* Overall document-level prediction: Displays averaged probabilities and the final class.

### Example Output
Document: Absences by Trump’s Senate pals help Democrats confirm Biden judges 

CNN Article, Left-leaning (https://adfontesmedia.com/interactive-media-bias-chart/) 

Probabilities Format: [Left, Center, Right]
```
Text: A ##bs ##ence ##s by Trump ' s Senate p ##als help Democrats confirm B...
Probabilities: [0.961, 0.021, 0.016]
Predicted Class: Left

Overall Predicted Class: Left
Overall Probabilities: [0.960, 0.012, 0.028]
```

### Error Handling
* If the specified file is not found, an error message is displayed.

### Usage
1. Set the path to your text file in the file_path variable.
2. Run the script:
```
python3 politicalbias.py
```
3. View the chunk-level and overall predictions in the console.
