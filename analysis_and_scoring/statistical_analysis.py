#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import math
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    df = pd.read_csv('per_document_metrics.csv')
    
    pipeline_mapping = {
        'baseline': 'Single-shot',
        'multi_agent': 'MAS', 
        'adaptive': 'Adaptive'
    }
    df['pipeline'] = df['pipeline'].map(pipeline_mapping)
    
    print("Data Overview:")
    print(f"Total documents: {len(df)}")
    print(f"Pipelines: {df['pipeline'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Complexity distribution: {df['complexity_label'].value_counts()}")
    
    key_metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1', 'latency_sec', 'token_usage']
    missing_data = df[key_metrics].isnull().sum()
    if missing_data.any():
        print(f"\nMissing data detected:")
        print(missing_data[missing_data > 0])
    
    print(f"\nSample distribution:")
    print(df.groupby(['pipeline', 'dataset']).size().unstack(fill_value=0))
    
    return df

def perform_kruskal_wallis_tests(df):
    print("\n" + "="*60)
    print("KRUSKAL-WALLIS TESTS (Overall Pipeline Comparison)")
    print("="*60)
    
    quality_metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    performance_metrics = ['latency_sec', 'token_usage', 'compression_ratio']
    
    results = {}
    
    for dataset in df['dataset'].unique():
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        
        dataset_df = df[df['dataset'] == dataset]
        
        single_shot = dataset_df[dataset_df['pipeline'] == 'Single-shot']
        mas = dataset_df[dataset_df['pipeline'] == 'MAS']
        adaptive = dataset_df[dataset_df['pipeline'] == 'Adaptive']
        
        results[dataset] = {}
        
        for metric in quality_metrics:
            if metric in dataset_df.columns:
                single_vals = single_shot[metric].dropna().values
                mas_vals = mas[metric].dropna().values
                adaptive_vals = adaptive[metric].dropna().values
                
                if len(single_vals) > 0 and len(mas_vals) > 0 and len(adaptive_vals) > 0:
                    h_stat, p_value = stats.kruskal(single_vals, mas_vals, adaptive_vals)
                    
                    n_total = len(single_vals) + len(mas_vals) + len(adaptive_vals)
                    eta_squared = (h_stat - 2) / (n_total - 3) if n_total > 3 else 0
                    
                    results[dataset][metric] = {
                        'h_stat': h_stat, 'p_value': p_value, 'eta_squared': eta_squared,
                        'n_single': len(single_vals), 'n_mas': len(mas_vals), 'n_adaptive': len(adaptive_vals),
                        'median_single': np.median(single_vals), 'median_mas': np.median(mas_vals), 'median_adaptive': np.median(adaptive_vals)
                    }
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"{metric:12}: H={h_stat:.3f}, p={p_value:.4f} {significance}, η²={eta_squared:.3f}")
                    print(f"             Medians: Single={np.median(single_vals):.3f}, MAS={np.median(mas_vals):.3f}, Adaptive={np.median(adaptive_vals):.3f}")
        
        for metric in performance_metrics:
            if metric in dataset_df.columns:
                single_vals = single_shot[metric].dropna().values
                mas_vals = mas[metric].dropna().values
                adaptive_vals = adaptive[metric].dropna().values
                
                if len(single_vals) > 0 and len(mas_vals) > 0 and len(adaptive_vals) > 0:
                    h_stat, p_value = stats.kruskal(single_vals, mas_vals, adaptive_vals)
                    
                    n_total = len(single_vals) + len(mas_vals) + len(adaptive_vals)
                    eta_squared = (h_stat - 2) / (n_total - 3) if n_total > 3 else 0
                    
                    results[dataset][metric] = {
                        'h_stat': h_stat, 'p_value': p_value, 'eta_squared': eta_squared,
                        'median_single': np.median(single_vals), 'median_mas': np.median(mas_vals), 'median_adaptive': np.median(adaptive_vals)
                    }
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"{metric:12}: H={h_stat:.3f}, p={p_value:.4f} {significance}, η²={eta_squared:.3f}")
                    print(f"             Medians: Single={np.median(single_vals):.3f}, MAS={np.median(mas_vals):.3f}, Adaptive={np.median(adaptive_vals):.3f}")
    
    return results

def perform_mann_whitney_tests(df):
    print("\n" + "="*60)
    print("MANN-WHITNEY U TESTS (Pairwise Comparisons)")
    print("="*60)
    
    pairs = [
        ('Single-shot', 'MAS'),
        ('Single-shot', 'Adaptive'),
        ('MAS', 'Adaptive')
    ]
    
    quality_metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    performance_metrics = ['latency_sec', 'token_usage']
    all_metrics = quality_metrics + performance_metrics
    
    total_comparisons = len(pairs) * len(df['dataset'].unique()) * len(all_metrics)
    alpha_bonferroni = 0.05 / total_comparisons
    print(f"Bonferroni correction: α = 0.05/{total_comparisons} = {alpha_bonferroni:.6f}")
    
    results = {}
    all_results_list = []  # For structured output
    
    for dataset in df['dataset'].unique():
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        
        dataset_df = df[df['dataset'] == dataset]
        results[dataset] = {}
        
        for pair in pairs:
            pipeline1, pipeline2 = pair
            print(f"\n{pipeline1} vs {pipeline2}")
            
            df1 = dataset_df[dataset_df['pipeline'] == pipeline1]
            df2 = dataset_df[dataset_df['pipeline'] == pipeline2]
            
            results[dataset][f"{pipeline1}_vs_{pipeline2}"] = {}
            
            for metric in all_metrics:
                if metric in dataset_df.columns and not df1.empty and not df2.empty:
                    vals1 = df1[metric].dropna().values
                    vals2 = df2[metric].dropna().values
                    
                    n1, n2 = len(vals1), len(vals2)
                    if n1 == 0 or n2 == 0:
                        print(f"  {metric:12}: SKIP - Empty groups (n1={n1}, n2={n2})")
                        continue
                    
                    if n1 + n2 != 150:
                        print(f"  {metric:12}: WARNING - Expected n1+n2=150, got {n1+n2} (n1={n1}, n2={n2})")
                    
                    u_stat, p_value = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    
                    mean_U = n1 * n2 / 2
                    sd_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    sd_U = max(sd_U, 1e-9)
                    
                    z = (u_stat - mean_U) / sd_U
                    effect_size_r = abs(z) / math.sqrt(n1 + n2)
                    
                    if not math.isfinite(effect_size_r):
                        effect_size_r = 1.0
                        print(f"  {metric:12}: WARNING - Infinite effect size, setting to 1.0")
                    
                    result_dict = {
                        'dataset': dataset,
                        'comparison': f"{pipeline1}_vs_{pipeline2}",
                        'metric': metric,
                        'u_stat': int(u_stat),
                        'p_value': p_value,
                        'p_bonferroni': min(p_value * total_comparisons, 1.0),
                        'effect_size_r': effect_size_r,
                        'z_score': z,
                        'median1': np.median(vals1),
                        'median2': np.median(vals2),
                        'n1': n1,
                        'n2': n2
                    }
                    
                    results[dataset][f"{pipeline1}_vs_{pipeline2}"][metric] = result_dict
                    all_results_list.append(result_dict)
                    
                    significance_raw = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    significance_adj = "***" if result_dict['p_bonferroni'] < 0.001 else "**" if result_dict['p_bonferroni'] < 0.01 else "*" if result_dict['p_bonferroni'] < 0.05 else "ns"
                    
                    median1, median2 = np.median(vals1), np.median(vals2)
                    direction = ">" if median1 > median2 else "<"
                    
                    print(f"  {metric:12}: U={int(u_stat)}, p={p_value:.4f} {significance_raw} (adj: {significance_adj}), r={effect_size_r:.3f}")
                    print(f"                ({median1:.3f} {direction} {median2:.3f}), n1={n1}, n2={n2}")
    
    results_df = pd.DataFrame(all_results_list)
    results_df.to_csv('outputs/mann_whitney_results.csv', index=False)
    print(f"\nDetailed results saved to outputs/mann_whitney_results.csv")
    
    print(f"\nUNIT TEST - Extreme Case Check:")
    try:
        extreme_test = results_df[
            (results_df['dataset'] == 'CNN/DM') & 
            (results_df['metric'] == 'latency_sec') & 
            (results_df['comparison'] == 'Single-shot_vs_MAS')
        ]
        if not extreme_test.empty:
            row = extreme_test.iloc[0]
            expected_r = math.sqrt(150) / 2
            print(f"  CNN/DM latency Single vs MAS: U={row['u_stat']}, r={row['effect_size_r']:.3f}")
            print(f"  Expected r ≈ {expected_r:.3f} for extreme case")
            if row['effect_size_r'] > 0.8:
                print(f"  Extreme case detected correctly")
            else:
                print(f"  Lower effect size than expected")
    except Exception as e:
        print(f"  Unit test failed: {e}")
    
    return results

def analyze_complexity_impact(df):
    print("\n" + "="*60)
    print("COMPLEXITY IMPACT ANALYSIS (RQ2)")
    print("="*60)
    
    print("\nLength-Based Complexity Distribution:")
    complexity_dist = df.groupby(['complexity_label', 'pipeline']).size().unstack(fill_value=0)
    print(complexity_dist)
    
    total_by_complexity = df['complexity_label'].value_counts()
    print(f"\nTotal documents per complexity level:")
    for complexity, count in total_by_complexity.items():
        percentage = (count / len(df)) * 100
        print(f"  {complexity}: {count} documents ({percentage:.1f}%)")
    
    quality_metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    efficiency_metrics = ['latency_sec', 'token_usage', 'compression_ratio']
    
    print(f"\nRQ2 ANALYSIS: Performance vs Complexity")
    print("=" * 50)
    
    for pipeline in df['pipeline'].unique():
        print(f"\nPipeline: {pipeline}")
        print("-" * 30)
        
        pipeline_df = df[df['pipeline'] == pipeline]
        
        simple_docs = pipeline_df[pipeline_df['complexity_label'] == 'simple']
        medium_docs = pipeline_df[pipeline_df['complexity_label'] == 'medium']
        complex_docs = pipeline_df[pipeline_df['complexity_label'] == 'complex']
        
        if len(simple_docs) > 0 and len(medium_docs) > 0 and len(complex_docs) > 0:
            print(f"Sample sizes: Simple={len(simple_docs)}, Medium={len(medium_docs)}, Complex={len(complex_docs)}")
            
            for metric in quality_metrics:
                if metric in pipeline_df.columns:
                    simple_vals = simple_docs[metric].dropna().values
                    medium_vals = medium_docs[metric].dropna().values
                    complex_vals = complex_docs[metric].dropna().values
                    
                    if len(simple_vals) > 0 and len(medium_vals) > 0 and len(complex_vals) > 0:
                        h_stat, p_value = stats.kruskal(simple_vals, medium_vals, complex_vals)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        
                        simple_med = np.median(simple_vals)
                        medium_med = np.median(medium_vals)
                        complex_med = np.median(complex_vals)
                        
                        trend = "↓" if simple_med > medium_med > complex_med else "↑" if simple_med < medium_med < complex_med else "~"
                        
                        print(f"  {metric:12}: H={h_stat:.3f}, p={p_value:.4f} {significance} {trend}")
                        print(f"                Simple={simple_med:.3f}, Medium={medium_med:.3f}, Complex={complex_med:.3f}")
            
            for metric in efficiency_metrics:
                if metric in pipeline_df.columns:
                    simple_vals = simple_docs[metric].dropna().values
                    medium_vals = medium_docs[metric].dropna().values
                    complex_vals = complex_docs[metric].dropna().values
                    
                    if len(simple_vals) > 0 and len(medium_vals) > 0 and len(complex_vals) > 0:
                        h_stat, p_value = stats.kruskal(simple_vals, medium_vals, complex_vals)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        
                        simple_med = np.median(simple_vals)
                        medium_med = np.median(medium_vals)
                        complex_med = np.median(complex_vals)
                        
                        trend = "↑" if simple_med < medium_med < complex_med else "↓" if simple_med > medium_med > complex_med else "~"
                        
                        print(f"  {metric:12}: H={h_stat:.3f}, p={p_value:.4f} {significance} {trend}")
                        print(f"                Simple={simple_med:.3f}, Medium={medium_med:.3f}, Complex={complex_med:.3f}")
    
    print(f"\nCROSS-PIPELINE COMPARISON BY COMPLEXITY LEVEL")
    print("=" * 50)
    
    for complexity in ['simple', 'medium', 'complex']:
        complexity_df = df[df['complexity_label'] == complexity]
        
        if len(complexity_df) > 0:
            print(f"\n{complexity.capitalize()} Documents (n={len(complexity_df)})")
            print("-" * 30)
            
            single_shot = complexity_df[complexity_df['pipeline'] == 'Single-shot']
            mas = complexity_df[complexity_df['pipeline'] == 'MAS']
            adaptive = complexity_df[complexity_df['pipeline'] == 'Adaptive']
            
            if len(single_shot) > 0 and len(mas) > 0 and len(adaptive) > 0:
                for metric in ['rouge1', 'rougeL', 'bertscore_f1']:
                    if metric in complexity_df.columns:
                        single_vals = single_shot[metric].dropna().values
                        mas_vals = mas[metric].dropna().values
                        adaptive_vals = adaptive[metric].dropna().values
                        
                        if len(single_vals) > 0 and len(mas_vals) > 0 and len(adaptive_vals) > 0:
                            h_stat, p_value = stats.kruskal(single_vals, mas_vals, adaptive_vals)
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            
                            print(f"  {metric:12}: H={h_stat:.3f}, p={p_value:.4f} {significance}")
                            print(f"                Single={np.median(single_vals):.3f}, MAS={np.median(mas_vals):.3f}, Adaptive={np.median(adaptive_vals):.3f}")
    
    return df

def generate_summary_report(kruskal_results, mann_whitney_results, df):
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    print("\nKEY FINDINGS:")
    
    print("\n1. OVERALL PIPELINE COMPARISON (Kruskal-Wallis):")
    significant_findings = 0
    
    for dataset in kruskal_results:
        print(f"\n   {dataset}:")
        for metric, result in kruskal_results[dataset].items():
            p_val = result['p_value']
            if p_val < 0.05:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                print(f"   ✓ {metric}: Significant difference (p={p_val:.4f}) {significance}")
                significant_findings += 1
            else:
                print(f"   ✗ {metric}: No significant difference (p={p_val:.4f})")
    
    print("\n2. BEST PERFORMING PIPELINE BY METRIC:")
    for dataset in df['dataset'].unique():
        print(f"\n   {dataset}:")
        dataset_df = df[df['dataset'] == dataset]
        
        for metric in ['rouge1', 'rougeL', 'bertscore_f1', 'latency_sec']:
            if metric in dataset_df.columns:
                best_pipeline = dataset_df.loc[dataset_df[metric].idxmax(), 'pipeline'] if metric != 'latency_sec' else dataset_df.loc[dataset_df[metric].idxmin(), 'pipeline']
                best_value = dataset_df[metric].max() if metric != 'latency_sec' else dataset_df[metric].min()
                print(f"   • {metric:12}: {best_pipeline} ({best_value:.3f})")
    
    print("\n3. PRACTICAL SIGNIFICANCE:")
    print("   Quality metrics (ROUGE-1, ROUGE-L, BERTScore):")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        single_shot = dataset_df[dataset_df['pipeline'] == 'Single-shot']
        mas = dataset_df[dataset_df['pipeline'] == 'MAS']
        adaptive = dataset_df[dataset_df['pipeline'] == 'Adaptive']
        
        if not single_shot.empty and not mas.empty and not adaptive.empty:
            rouge1_range = dataset_df['rouge1'].max() - dataset_df['rouge1'].min()
            bert_range = dataset_df['bertscore_f1'].max() - dataset_df['bertscore_f1'].min()
            time_range = dataset_df['latency_sec'].max() - dataset_df['latency_sec'].min()
            
            print(f"   • {dataset}: ROUGE-1 range={rouge1_range:.3f}, BERTScore range={bert_range:.3f}, Time range={time_range:.1f}s")
    
    print(f"\nSTATISTICAL POWER:")
    print(f"   • Total significant findings: {significant_findings}")
    print(f"   • Sample size per pipeline: ~75 examples")
    print(f"   • Statistical tests used: Kruskal-Wallis + Mann-Whitney U")
    print(f"   • Significance level: α = 0.05")
    
    print("\nRECOMMENDATIONS FOR THESIS:")
    print("   1. Focus on datasets with significant differences")
    print("   2. Report effect sizes alongside p-values")
    print("   3. Consider practical significance (time vs quality trade-offs)")
    print("   4. Use complexity analysis to explain performance differences")

def create_visualization_plots(df):
    print("\nCreating visualization plots...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: ROUGE-1 by Pipeline and Dataset
    sns.boxplot(data=df, x='dataset', y='rouge1', hue='pipeline', ax=axes[0,0])
    axes[0,0].set_title('ROUGE-1 Scores by Pipeline and Dataset')
    axes[0,0].set_ylabel('ROUGE-1 Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: BERTScore by Pipeline and Dataset
    sns.boxplot(data=df, x='dataset', y='bertscore_f1', hue='pipeline', ax=axes[0,1])
    axes[0,1].set_title('BERTScore F1 by Pipeline and Dataset')
    axes[0,1].set_ylabel('BERTScore F1')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Processing Time by Pipeline and Dataset
    sns.boxplot(data=df, x='dataset', y='latency_sec', hue='pipeline', ax=axes[0,2])
    axes[0,2].set_title('Processing Time by Pipeline and Dataset')
    axes[0,2].set_ylabel('Processing Time (seconds)')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Quality vs Time Trade-off
    sns.scatterplot(data=df, x='latency_sec', y='rouge1', hue='pipeline', style='dataset', s=50, ax=axes[1,0])
    axes[1,0].set_title('Quality vs Speed Trade-off')
    axes[1,0].set_xlabel('Processing Time (seconds)')
    axes[1,0].set_ylabel('ROUGE-1 Score')
    
    # Plot 5: Performance by Complexity Level
    sns.boxplot(data=df, x='complexity_label', y='rouge1', hue='pipeline', ax=axes[1,1])
    axes[1,1].set_title('ROUGE-1 by Complexity Level')
    axes[1,1].set_ylabel('ROUGE-1 Score')
    axes[1,1].set_xlabel('Document Complexity')
    
    # Plot 6: Token Usage by Complexity
    sns.boxplot(data=df, x='complexity_label', y='token_usage', hue='pipeline', ax=axes[1,2])
    axes[1,2].set_title('Token Usage by Complexity Level')
    axes[1,2].set_ylabel('Token Usage')
    axes[1,2].set_xlabel('Document Complexity')
    
    plt.tight_layout()
    plt.savefig('outputs/statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved to outputs/statistical_analysis_plots.png")
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot complexity distributions
    complexity_counts = df['complexity_label'].value_counts()
    axes2[0].pie(complexity_counts.values, labels=complexity_counts.index, autopct='%1.1f%%')
    axes2[0].set_title('Overall Complexity Distribution')
    
    # Average word count by complexity
    word_count_by_complexity = df.groupby('complexity_label')['word_count'].mean()
    axes2[1].bar(word_count_by_complexity.index, word_count_by_complexity.values)
    axes2[1].set_title('Average Word Count by Complexity')
    axes2[1].set_ylabel('Word Count')
    axes2[1].set_xlabel('Complexity Level')
    
    # Performance degradation with complexity
    quality_by_complexity = df.groupby(['complexity_label', 'pipeline'])['rouge1'].mean().unstack()
    quality_by_complexity.plot(kind='bar', ax=axes2[2])
    axes2[2].set_title('ROUGE-1 Performance by Complexity')
    axes2[2].set_ylabel('Average ROUGE-1')
    axes2[2].set_xlabel('Complexity Level')
    axes2[2].tick_params(axis='x', rotation=0)
    axes2[2].legend(title='Pipeline')
    
    plt.tight_layout()
    plt.savefig('outputs/complexity_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Complexity plots saved to outputs/complexity_analysis_plots.png")

def main():
    print("STATISTICAL ANALYSIS FOR THESIS")
    print("Comparing Single-shot vs MAS vs Adaptive pipelines")
    print("="*60)
    
    df = load_and_prepare_data()
    
    kruskal_results = perform_kruskal_wallis_tests(df)
    mann_whitney_results = perform_mann_whitney_tests(df)
    
    analyze_complexity_impact(df)
    
    generate_summary_report(kruskal_results, mann_whitney_results, df)
    
    create_visualization_plots(df)
    
    print("\nStatistical analysis complete!")
    print("Results ready for thesis writeup")

if __name__ == "__main__":
    main()