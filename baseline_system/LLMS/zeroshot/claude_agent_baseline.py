import os
import json
import anthropic
import time
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.3

def generate_summary(text):
    max_chars = 100000
    
    if len(text) > max_chars:
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}")
            prompt = f"""
            Summarize the following text concisely:

            {chunk}

            Summary:
            """
            
            response = client.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=500
            )
            chunk_summaries.append(response.content[0].text.strip())
            time.sleep(1)
        
        combined_text = "\n\n".join(chunk_summaries)
        prompt = f"""
        Create a comprehensive summary from these partial summaries:

        {combined_text}

        Final Summary:
        """
    else:
        prompt = f"""
        Summarize the following text concisely:

        {text}

        Summary:
        """

    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=1000
    )
    return response.content[0].text.strip()

def calculate_complexity_analysis(text):
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    unique_words = len(set(word.lower() for word in words))
    lexical_diversity = unique_words / max(word_count, 1)
    
    length_complexity = min(word_count / 2000, 1.0)
    structure_complexity = min(avg_sentence_length / 35, 1.0)
    diversity_complexity = lexical_diversity
    
    complexity_score = (length_complexity + structure_complexity + diversity_complexity) / 3
    
    if complexity_score < 0.33:
        complexity_category = "simple"
    elif complexity_score < 0.67:
        complexity_category = "medium"
    else:
        complexity_category = "complex"
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": len(paragraphs),
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "complexity_score": complexity_score,
        "complexity_category": complexity_category
    }

def calculate_compression_ratio(source_text, summary_text):
    source_words = len(source_text.split())
    summary_words = len(summary_text.split())
    return source_words / max(summary_words, 1)

def calculate_repetition_score(text):
    if not text.strip():
        return 0.0
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    repeated = sum(count - 1 for count in bigram_counts.values() if count > 1)
    return repeated / max(len(bigrams), 1)

def estimate_tokens(text):
    return int(len(text.split()) / 0.75)

def get_value(row, keys):
    for k in keys:
        if k in row:
            return row[k]
    raise KeyError(f"None of the keys {keys} found in row: {row}")

def summarize_dataset(input_path, output_path, doc_key, sum_key, start=0, end=None, delay=3):
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if end is None:
        end = len(data)

    data = data[start:end]

    results = []
    for i, row in enumerate(data, start=start):
        document = row[doc_key]
        reference = row[sum_key]
        
        start_time = time.time()
        summary = generate_summary(document)
        processing_time = time.time() - start_time
        
        complexity_analysis = calculate_complexity_analysis(document)
        compression_ratio = calculate_compression_ratio(document, summary)
        repetition_score = calculate_repetition_score(summary)
        tokens = estimate_tokens(document)
        
        results.append({
            "id": i,
            "source": document,
            "reference": reference,
            "generated": summary,
            "method": "single_shot_baseline",
            "processing_time": processing_time,
            "complexity_analysis": complexity_analysis,
            "compression_ratio": compression_ratio,
            "repetition_score": repetition_score,
            "tokens": tokens,
        })
        print(f"[{i-start+1}/{end-start}] Done")
        
        time.sleep(delay)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    datasets = [
        ("data/xsum_dev.jsonl", "outputs/no-tools/xsum_claude.json", "document", "summary"),
        ("data/cnn_dailymail_dev.jsonl", "outputs/no-tools/cnn_dailymail_claude.json", "article", "highlights"),
        ("data/govreport_dev.jsonl", "outputs/no-tools/govreport_claude.json", "report", "summary")
    ]
    
    for input_path, output_path, doc_key, sum_key in datasets:
        print(f"Processing {input_path} (first 75 examples)...")
        summarize_dataset(input_path, output_path, doc_key, sum_key, end=75, delay=3)