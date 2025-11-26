from sentence_transformers import CrossEncoder
from typing import List
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize the model (will be loaded on first use)
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading the language model (this will be faster and use less memory)...")
        # Using a small cross-encoder for better quality than just keyword matching
        _model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)
    return _model

def generate_answer(query: str, retrieved_texts: List[str]) -> str:
    """
    Generate an answer using a local LLM based on the retrieved context.
    
    Args:
        query: The user's question
        retrieved_texts: List of relevant text chunks from the knowledge base
        
    Returns:
        str: Generated answer
    """
    # Combine the retrieved texts into a single context
    context = "\n\n".join(retrieved_texts)
    
    # Create the prompt
    prompt = f"""### System:
    You are an AI assistant that answers questions based on the provided context.
    Use only the information from the context to answer the question.
    If the context doesn't contain the answer, say "I don't have enough information to answer this question."
    
    ### Context:
    {context}
    
    ### Question:
    {query}
    
    ### Answer:
    """
    
    try:
        # Get the model
        model = get_model()
        
        # Use cross-encoder to find the most relevant sentence
        sentences = []
        for text in retrieved_texts:
            sentences.extend([s.strip() for s in re.split(r'[.!?]', text) if len(s.split()) > 3])
        
        if not sentences:
            return "I couldn't find enough information to answer this question."
            
        # Score each sentence
        pairs = [[query, sent] for sent in sentences]
        scores = model.predict(pairs)
        
        # Get the best sentence
        best_idx = scores.argmax()
        best_sentence = sentences[best_idx].strip()
        
        # Simple answer formatting
        answer = f"Based on the information I found: {best_sentence}"
        
        return answer
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        # Fallback to simple extractive method if LLM fails
        query_words = set(query.lower().split())
        best_para = ""
        best_score = 0

        for text in retrieved_texts:
            paragraphs = text.split("\n")
            for para in paragraphs:
                score = len(set(para.lower().split()) & query_words)
                if score > best_score:
                    best_score = score
                    best_para = para
                    
        return best_para if best_para else "I couldn't generate an answer for this question."
    return best_para if best_para else "Answer not available in KB."