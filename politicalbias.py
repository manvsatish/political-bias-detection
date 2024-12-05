from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
import torch

logging.set_verbosity_error()

# Function to classify text and return predictions
def classify_political_bias(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
    
    # Split text into chunks of 512 tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get model outputs for each chunk
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Map the predictions to classes
    class_names = ["Left", "Center", "Right"]
    predicted_class = torch.argmax(probs).item()
    
    print(f"Text: {text[:100]}...") #readabilitu
    print(f"Probabilities: {probs.tolist()}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    return class_names[predicted_class], probs.tolist()


if __name__ == "__main__":
    file_path = "documents/story_of_how_trump.txt" 

    try:
        with open(file_path, "r") as file:
            text = file.read().strip()
            
        # If the text is too long, split it into manageable chunks
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokens = tokenizer.tokenize(text)
        chunk_size = 512
        chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        
        # Classify each chunk and average the probabilities
        total_probs = torch.zeros(3)  # Initialize a tensor for the three classes
        for chunk in chunks:
            predicted_class, probabilities = classify_political_bias(chunk)
            total_probs += torch.tensor(probabilities)
        
        # Average probabilities over all chunks
        averaged_probs = total_probs / len(chunks)
        overall_class = torch.argmax(averaged_probs).item()
        class_names = ["Left", "Center", "Right"]
        
        print(f"\nOverall Predicted Class: {class_names[overall_class]}")
        print(f"Overall Probabilities: {averaged_probs.tolist()}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
