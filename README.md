# Text Summarization Systems Evaluation

Name - Aryan Sharma (2774404)

This repository contains three text summarization systems and their complete evaluation pipeline.

## Systems

### 1. Baseline System
**Location**: `baseline_system/LLMS/zeroshot/`
- `claude_agent_baseline.py` - Claude-based baseline summarization

### 2. Real MAS System  
**Location**: `real_mas_system/`
- `agents/` - Multi-agent system components
- `run_real_mas_system.py` - Main execution script

### 3. Real Adaptive System
**Location**: `real_adaptive_system/`
- `run_real_adaptive_system.py` - Main execution script

## Data
**Location**: `data/`
- `cnn_dailymail_dev.jsonl`
- `govreport_dev.jsonl`  
- `xsum_dev.jsonl`

## Running the Systems

### Baseline System
```bash
cd baseline_system/LLMS/zeroshot/
python claude_agent_baseline.py
```

### Real MAS System
```bash
cd real_mas_system/
python run_real_mas_system.py
```

### Real Adaptive System
```bash
cd real_adaptive_system/
python run_real_adaptive_system.py
```

## Complete Evaluation Pipeline

**Location**: `analysis_and_scoring/`

Run these 3 scripts in order for complete evaluation:

### 1. Generate Master Dataset with ROUGE & BERT Scores
```bash
python create_per_document_table_with_real_systems.py
```
- Calculates ROUGE and BERT scores for all three systems
- Creates comprehensive per-document metrics dataset
- Processes all datasets (XSum, CNN/DailyMail, GovReport)

### 2. Statistical Analysis
```bash
python statistical_analysis.py
```
- Performs Kruskal-Wallis and Mann-Whitney U tests
- Complexity-stratified analysis
- Effect size calculations with Bonferroni corrections

### 3. Generate Publication Plots
```bash
python create_thesis_plots.py
```
- Creates publication-quality visualizations
- Complexity analysis plots
- Statistical comparison plots

## Results
**Location**: `outputs/`
- `real_mas/` - Multi-agent system results and checkpoints
- `real_adaptive/` - Adaptive system results and checkpoints  
- `no-tools/` - Baseline system results
- Statistical analysis CSV files and plots

