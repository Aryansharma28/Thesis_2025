#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path
import re

try:
    import evaluate
    from datasets import Dataset
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Error loading evaluation libraries: {e}")
    METRICS_AVAILABLE = False

def calculate_rouge_bertscore(predictions, references):
    if not METRICS_AVAILABLE:
        return {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0
        }
    
    try:
        rouge_results = rouge.compute(predictions=predictions, references=references)
        bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
        
        return {
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'], 
            'rougeL': rouge_results['rougeL'],
            'bertscore_precision': sum(bertscore_results['precision']) / len(bertscore_results['precision']),
            'bertscore_recall': sum(bertscore_results['recall']) / len(bertscore_results['recall']),
            'bertscore_f1': sum(bertscore_results['f1']) / len(bertscore_results['f1'])
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0
        }

def extract_dataset_from_filename(filename):
    if 'xsum' in filename.lower():
        return 'XSum'
    elif 'cnn_dailymail' in filename.lower():
        return 'CNN/DM'
    elif 'govreport' in filename.lower():
        return 'GovReport'
    else:
        return 'Unknown'

def apply_length_based_complexity(word_count):
    if word_count <= 416:
        return "simple"
    elif word_count <= 1539:
        return "medium" 
    else:
        return "complex"

def extract_metrics_from_file(file_path, pipeline_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a list")
            return []
        
        dataset_name = extract_dataset_from_filename(file_path)
        
        predictions = []
        references = []
        
        for item in data:
            if 'generated' in item and 'reference' in item:
                predictions.append(str(item['generated']))
                references.append(str(item['reference']))
        
        batch_metrics = calculate_rouge_bertscore(predictions, references)
        
        rows = []
        for i, item in enumerate(data):
            try:
                if not all(key in item for key in ['id', 'source', 'reference', 'generated']):
                    continue
                
                complexity_analysis = item.get('complexity_analysis', {})
                word_count = complexity_analysis.get('word_count', len(item['source'].split()))
                complexity_label = apply_length_based_complexity(word_count)
                
                processing_time = item.get('processing_time', 0)
                tokens = item.get('tokens', 0)
                compression_ratio = item.get('compression_ratio', 0)
                repetition_score = item.get('repetition_score', 0)
                
                if i < len(predictions):
                    rouge1 = batch_metrics['rouge1']
                    rouge2 = batch_metrics['rouge2']
                    rougeL = batch_metrics['rougeL']
                    bertscore_f1 = batch_metrics['bertscore_f1']
                else:
                    rouge1 = rouge2 = rougeL = bertscore_f1 = 0.0
                
                adaptive_metadata = item.get('adaptive_metadata', {})
                mas_metadata = item.get('mas_metadata', {})
                
                feedback_iterations = 0
                if adaptive_metadata:
                    feedback_iterations = adaptive_metadata.get('feedback_iterations', 0)
                
                row = {
                    'document_id': item['id'],
                    'dataset': dataset_name,
                    'pipeline': pipeline_name,
                    'source_text': item['source'],
                    'reference_summary': item['reference'],
                    'generated_summary': item['generated'],
                    'word_count': word_count,
                    'complexity_label': complexity_label,
                    'rouge1': rouge1,
                    'rouge2': rouge2,
                    'rougeL': rougeL,
                    'bertscore_f1': bertscore_f1,
                    'compression_ratio': compression_ratio,
                    'repetition_score': repetition_score,
                    'latency_sec': processing_time,
                    'token_usage': tokens,
                    'feedback_iterations': feedback_iterations,
                    'method': item.get('method', pipeline_name)
                }
                
                rows.append(row)
                
            except Exception as e:
                print(f"Error processing item {i} in {file_path}: {e}")
                continue
        
        return rows
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def create_comprehensive_table():
    print("Creating comprehensive per-document metrics table...")
    
    all_rows = []
    
    file_mappings = [
        ("outputs/no-tools/xsum_claude.json", "baseline"),
        ("outputs/no-tools/cnn_dailymail_claude.json", "baseline"),
        ("outputs/no-tools/govreport_claude.json", "baseline"),
        ("outputs/enhanced_claude/xsum_multi_agent_enhanced.json", "enhanced_multi_agent"),
        ("outputs/enhanced_claude/cnn_dailymail_multi_agent_enhanced.json", "enhanced_multi_agent"),
        ("outputs/enhanced_claude/govreport_multi_agent_enhanced.json", "enhanced_multi_agent"),
        ("outputs/enhanced_claude/xsum_adaptive_enhanced.json", "enhanced_adaptive"),
        ("outputs/enhanced_claude/cnn_dailymail_adaptive_enhanced.json", "enhanced_adaptive"),
        ("outputs/enhanced_claude/govreport_adaptive_enhanced.json", "enhanced_adaptive"),
        ("outputs/real_mas/xsum_real_mas.json", "real_mas"),
        ("outputs/real_mas/cnn_dailymail_real_mas.json", "real_mas"),
        ("outputs/real_mas/govreport_real_mas.json", "real_mas"),
        ("outputs/real_adaptive/xsum_real_adaptive.json", "real_adaptive"),
        ("outputs/real_adaptive/cnn_dailymail_real_adaptive.json", "real_adaptive"),
        ("outputs/real_adaptive/govreport_real_adaptive.json", "real_adaptive"),
    ]
    
    for file_path, pipeline_name in file_mappings:
        if Path(file_path).exists():
            print(f"Processing {file_path} ({pipeline_name})...")
            rows = extract_metrics_from_file(file_path, pipeline_name)
            all_rows.extend(rows)
            print(f"  Added {len(rows)} documents")
        else:
            print(f"Warning: {file_path} not found")
    
    if not all_rows:
        print("No data found in any files")
        return
    
    df = pd.DataFrame(all_rows)
    df = df.sort_values(['dataset', 'pipeline', 'document_id']).reset_index(drop=True)
    
    output_file = "per_document_metrics_with_real_systems.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nComprehensive table created: {output_file}")
    print(f"Total documents: {len(df)}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Pipelines: {df['pipeline'].unique()}")
    
    print(f"\nDocuments per pipeline:")
    pipeline_counts = df['pipeline'].value_counts()
    for pipeline, count in pipeline_counts.items():
        print(f"  {pipeline}: {count} documents")
    
    print(f"\nDocuments per dataset:")
    dataset_counts = df.groupby(['dataset', 'pipeline']).size().unstack(fill_value=0)
    print(dataset_counts)
    
    return df

def main():
    print("COMPREHENSIVE PER-DOCUMENT TABLE CREATOR")
    print("=" * 60)
    print("Including: Baseline + Enhanced Simulations + Real MAS + Real Adaptive")
    print("=" * 60)
    
    df = create_comprehensive_table()
    
    if df is not None:
        print(f"\nSuccess! Table ready for statistical analysis.")
        print(f"Use 'per_document_metrics_with_real_systems.csv' for analysis")
        
        print(f"\nSample of the data:")
        print(df[['dataset', 'pipeline', 'rouge1', 'bertscore_f1', 'latency_sec', 'feedback_iterations']].head(10))

if __name__ == "__main__":
    main()