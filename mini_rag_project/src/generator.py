def generate_answer(query, retrieved_texts):
    """
    Simple extractive answering.
    Chooses paragraph with highest keyword overlap.
    """
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

    return best_para if best_para else "Answer not available in KB."