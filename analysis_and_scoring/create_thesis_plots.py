#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_data():
    try:
        df = pd.read_csv('per_document_metrics.csv')
        print(f"Loaded {len(df)} documents from per_document_metrics.csv")
        return df
    except FileNotFoundError:
        print("per_document_metrics.csv not found!")
        return None

def filter_to_target_pipelines(df):
    target_pipelines = ['claude_baseline', 'real_mas', 'real_adaptive']
    
    available_pipelines = df['pipeline'].unique()
    print(f"Available pipelines: {available_pipelines}")
    
    filtered_df = df[df['pipeline'].isin(target_pipelines)].copy()
    
    if len(filtered_df) == 0:
        print("No data found for target pipelines!")
        return None
    
    pipeline_mapping = {
        'claude_baseline': 'Single-shot',
        'real_mas': 'MAS',
        'real_adaptive': 'Adaptive'
    }
    filtered_df['pipeline'] = filtered_df['pipeline'].map(pipeline_mapping)
    
    print(f"Filtered to {len(filtered_df)} documents across {filtered_df['pipeline'].nunique()} pipelines")
    return filtered_df

def create_complexity_analysis_plot(df):
    """
    Create Plot 1: Complexity Analysis
    Shows how pipeline performance varies across document complexity levels
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Plot 1: Performance Analysis by Document Complexity', fontsize=16, fontweight='bold')
    
    # Color palette for consistency
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    # (a) Complexity Distribution
    ax = axes[0, 0]
    complexity_counts = df['complexity_label'].value_counts().sort_index()
    bars = ax.bar(complexity_counts.index, complexity_counts.values, color=colors[0], alpha=0.7)
    ax.set_title('(a) Document Complexity Distribution', fontweight='bold')
    ax.set_ylabel('Number of Documents')
    ax.set_xlabel('Complexity Level')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    # (b) ROUGE-1 by Complexity
    ax = axes[0, 1]
    sns.boxplot(data=df, x='complexity_label', y='rouge1', hue='pipeline', 
                palette=colors, ax=ax)
    ax.set_title('(b) ROUGE-1 Performance by Complexity', fontweight='bold')
    ax.set_ylabel('ROUGE-1 Score')
    ax.set_xlabel('Complexity Level')
    ax.legend(title='Pipeline', loc='upper right')
    
    # (c) BERTScore by Complexity
    ax = axes[0, 2]
    sns.boxplot(data=df, x='complexity_label', y='bertscore_f1', hue='pipeline',
                palette=colors, ax=ax)
    ax.set_title('(c) BERTScore F1 by Complexity', fontweight='bold')
    ax.set_ylabel('BERTScore F1')
    ax.set_xlabel('Complexity Level')
    ax.legend(title='Pipeline', loc='upper right')
    
    # (d) Processing Time by Complexity
    ax = axes[1, 0]
    sns.boxplot(data=df, x='complexity_label', y='latency', hue='pipeline',
                palette=colors, ax=ax)
    ax.set_title('(d) Processing Time by Complexity', fontweight='bold')
    ax.set_ylabel('Processing Time (seconds)')
    ax.set_xlabel('Complexity Level')
    ax.legend(title='Pipeline')
    
    # (e) Token Usage by Complexity
    ax = axes[1, 1]
    sns.boxplot(data=df, x='complexity_label', y='token_usage', hue='pipeline',
                palette=colors, ax=ax)
    ax.set_title('(e) Token Usage by Complexity', fontweight='bold')
    ax.set_ylabel('Token Usage')
    ax.set_xlabel('Complexity Level')
    ax.legend(title='Pipeline')
    
    # (f) Performance Degradation Trends
    ax = axes[1, 2]
    
    # Calculate mean performance by complexity for each pipeline
    perf_by_complexity = df.groupby(['complexity_label', 'pipeline'])['rouge1'].mean().reset_index()
    complexity_order = ['simple', 'medium', 'complex']
    
    for i, pipeline in enumerate(['Single-shot', 'MAS', 'Adaptive']):
        pipeline_data = perf_by_complexity[perf_by_complexity['pipeline'] == pipeline]
        if not pipeline_data.empty:
            # Ensure proper ordering
            pipeline_data = pipeline_data.set_index('complexity_label').reindex(complexity_order)
            ax.plot(complexity_order, pipeline_data['rouge1'], 
                   marker='o', linewidth=2, markersize=8, color=colors[i], label=pipeline)
    
    ax.set_title('(f) Performance Degradation Trends', fontweight='bold')
    ax.set_ylabel('Mean ROUGE-1 Score')
    ax.set_xlabel('Complexity Level')
    ax.legend(title='Pipeline')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save plot
    output_path = 'outputs/plot1_complexity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot 1 saved to {output_path}")
    plt.show()

def calculate_statistical_significance(df):
    results = {}
    
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1', 'latency', 'token_usage']
    
    for dataset in df['dataset'].unique():
        results[dataset] = {}
        dataset_df = df[df['dataset'] == dataset]
        
        for metric in metrics:
            if metric in dataset_df.columns:
                single_shot = dataset_df[dataset_df['pipeline'] == 'Single-shot'][metric].dropna()
                mas = dataset_df[dataset_df['pipeline'] == 'MAS'][metric].dropna()
                adaptive = dataset_df[dataset_df['pipeline'] == 'Adaptive'][metric].dropna()
                
                if len(single_shot) > 0 and len(mas) > 0 and len(adaptive) > 0:
                    h_stat, p_value = stats.kruskal(single_shot, mas, adaptive)
                    
                    n_total = len(single_shot) + len(mas) + len(adaptive)
                    eta_squared = (h_stat - 2) / (n_total - 3) if n_total > 3 else 0
                    
                    results[dataset][metric] = {
                        'h_stat': h_stat,
                        'p_value': p_value,
                        'eta_squared': eta_squared,
                        'significant': p_value < 0.05,
                        'medians': {
                            'Single-shot': single_shot.median(),
                            'MAS': mas.median(),
                            'Adaptive': adaptive.median()
                        }
                    }
    
    return results

def create_statistical_analysis_plot(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Plot 2: Statistical Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    print("Calculating statistical significance...")
    sig_results = calculate_statistical_significance(df)
    
    ax = axes[0, 0]
    sns.boxplot(data=df, x='dataset', y='rouge1', hue='pipeline', 
                palette=colors, ax=ax)
    ax.set_title('(a) ROUGE-1 Performance by Dataset', fontweight='bold')
    ax.set_ylabel('ROUGE-1 Score')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Pipeline')
    
    for i, dataset in enumerate(df['dataset'].unique()):
        if dataset in sig_results and 'rouge1' in sig_results[dataset]:
            p_val = sig_results[dataset]['rouge1']['p_value']
            if p_val < 0.05:
                sig_symbol = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                ax.text(i, ax.get_ylim()[1] * 0.95, sig_symbol, ha='center', va='top', 
                       fontsize=16, fontweight='bold')
    
    # (b) BERTScore Comparison by Dataset
    ax = axes[0, 1]
    sns.boxplot(data=df, x='dataset', y='bertscore_f1', hue='pipeline',
                palette=colors, ax=ax)
    ax.set_title('(b) BERTScore F1 by Dataset', fontweight='bold')
    ax.set_ylabel('BERTScore F1')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Pipeline')
    
    # (c) Processing Time Comparison
    ax = axes[0, 2]
    sns.boxplot(data=df, x='dataset', y='latency', hue='pipeline',
                palette=colors, ax=ax)
    ax.set_title('(c) Processing Time by Dataset', fontweight='bold')
    ax.set_ylabel('Processing Time (seconds)')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Pipeline')
    
    # (d) Quality vs Speed Trade-off
    ax = axes[1, 0]
    
    # Create scatter plot with different markers for datasets
    markers = ['o', 's', '^']  # circle, square, triangle
    dataset_names = df['dataset'].unique()
    
    for i, pipeline in enumerate(['Single-shot', 'MAS', 'Adaptive']):
        pipeline_data = df[df['pipeline'] == pipeline]
        
        for j, dataset in enumerate(dataset_names):
            dataset_pipeline_data = pipeline_data[pipeline_data['dataset'] == dataset]
            if not dataset_pipeline_data.empty:
                ax.scatter(dataset_pipeline_data['latency'], dataset_pipeline_data['rouge1'],
                          color=colors[i], marker=markers[j], alpha=0.7, s=50,
                          label=f'{pipeline}-{dataset}' if i == 0 else "")
    
    ax.set_title('(d) Quality vs Speed Trade-off', fontweight='bold')
    ax.set_xlabel('Processing Time (seconds)')
    ax.set_ylabel('ROUGE-1 Score')
    
    # Create custom legend
    pipeline_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                  markersize=8, label=pipeline) 
                       for i, pipeline in enumerate(['Single-shot', 'MAS', 'Adaptive'])]
    dataset_handles = [plt.Line2D([0], [0], marker=markers[i], color='gray', 
                                 markersize=8, label=dataset)
                      for i, dataset in enumerate(dataset_names)]
    
    legend1 = ax.legend(handles=pipeline_handles, title='Pipeline', loc='upper left')
    legend2 = ax.legend(handles=dataset_handles, title='Dataset', loc='lower right')
    ax.add_artist(legend1)
    
    # (e) Token Usage Comparison
    ax = axes[1, 1]
    sns.boxplot(data=df, x='dataset', y='token_usage', hue='pipeline',
                palette=colors, ax=ax)
    ax.set_title('(e) Token Usage by Dataset', fontweight='bold')
    ax.set_ylabel('Token Usage')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Pipeline')
    
    # (f) Effect Sizes Heatmap
    ax = axes[1, 2]
    
    # Create effect size matrix
    effect_size_data = []
    datasets = df['dataset'].unique()
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    
    for dataset in datasets:
        for metric in metrics:
            if dataset in sig_results and metric in sig_results[dataset]:
                eta_squared = sig_results[dataset][metric]['eta_squared']
                effect_size_data.append({
                    'Dataset': dataset,
                    'Metric': metric.upper().replace('_', '-'),
                    'Effect Size': eta_squared
                })
    
    if effect_size_data:
        effect_df = pd.DataFrame(effect_size_data)
        effect_pivot = effect_df.pivot(index='Dataset', columns='Metric', values='Effect Size')
        
        sns.heatmap(effect_pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Effect Size (η²)'})
        ax.set_title('(f) Statistical Effect Sizes', fontweight='bold')
        ax.set_xlabel('Quality Metrics')
        ax.set_ylabel('Dataset')
    else:
        ax.text(0.5, 0.5, 'No significant\neffects found', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(f) Statistical Effect Sizes', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    output_path = 'outputs/plot2_statistical_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot 2 saved to {output_path}")
    plt.show()
    
    return sig_results

def print_statistical_summary(sig_results):
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*60)
    
    significant_count = 0
    total_tests = 0
    
    for dataset in sig_results:
        print(f"\n{dataset}:")
        for metric, result in sig_results[dataset].items():
            total_tests += 1
            if result['significant']:
                significant_count += 1
                print(f"  ✓ {metric}: H={result['h_stat']:.3f}, p={result['p_value']:.4f}, η²={result['eta_squared']:.3f}")
                
                medians = result['medians']
                best_pipeline = max(medians.keys(), key=lambda k: medians[k])
                print(f"    Best: {best_pipeline} ({medians[best_pipeline]:.3f})")
            else:
                print(f"  ✗ {metric}: No significant difference (p={result['p_value']:.4f})")
    
    print(f"\nOverall: {significant_count}/{total_tests} tests showed significance (α=0.05)")

def main():
    print("CREATING THESIS PLOTS")
    print("="*60)
    print("Plot 1: Complexity Analysis")
    print("Plot 2: Statistical Analysis")
    print("Pipelines: claude_baseline → Single-shot, real_mas → MAS, real_adaptive → Adaptive")
    print("="*60)
    
    Path('outputs').mkdir(exist_ok=True)
    
    df = load_data()
    if df is None:
        return
    
    df = filter_to_target_pipelines(df)
    if df is None:
        return
    
    print(f"\nData Summary:")
    print(f"  - Total documents: {len(df)}")
    print(f"  - Pipelines: {df['pipeline'].unique()}")
    print(f"  - Datasets: {df['dataset'].unique()}")
    print(f"  - Complexity levels: {df['complexity_label'].unique()}")
    
    print(f"\nCreating Plot 1: Complexity Analysis...")
    create_complexity_analysis_plot(df)
    
    print(f"\nCreating Plot 2: Statistical Analysis...")
    sig_results = create_statistical_analysis_plot(df)
    
    print_statistical_summary(sig_results)
    
    print(f"\nBoth plots created successfully!")
    print(f"Saved to:")
    print(f"  - outputs/plot1_complexity_analysis.png")
    print(f"  - outputs/plot2_statistical_analysis.png")

if __name__ == "__main__":
    main()