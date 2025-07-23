import json
import evaluate

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def evaluate(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    references = [r["reference"] for r in results]
    predictions = [r["generated"] for r in results]

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

    print("\n=== ROUGE ===")
    for k, v in rouge_scores.items():
        print(f"{k}: {v:.4f}")

    print("\n=== BERTScore ===")
    print(f"Precision: {sum(bert_scores['precision']) / len(bert_scores['precision']):.4f}")
    print(f"Recall:    {sum(bert_scores['recall']) / len(bert_scores['recall']):.4f}")
    print(f"F1:        {sum(bert_scores['f1']) / len(bert_scores['f1']):.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate_outputs.py <output_file.json>")
    else:
        evaluate(sys.argv[1])
