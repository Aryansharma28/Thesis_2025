#!/usr/bin/env python3

import os
import json
import time
import sys
import re
from pathlib import Path
from collections import Counter
import gc

# Add agents directory to path
sys.path.append('agents')

# Load environment manually
def load_env_file():
        env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    os.environ[key] = value

load_env_file()

try:
    from agents.macro_agent.macro_agent import MacroAgent
    from agents.macro_agent.utils.config import MASConfig, AgentConfig
    from agents.macro_agent.utils.logger import MASLogger
    MAS_AVAILABLE = True
    print("✅ Real MAS system imported successfully")
except ImportError as e:
    print(f"❌ Failed to import real MAS system: {e}")
    MAS_AVAILABLE = False

BASE_DELAY = 3.0
MAX_CONSECUTIVE_FAILURES = 5
MAX_RETRIES = 3

class AdvancedMetrics:
    
    @staticmethod
    def calculate_compression_ratio(source_text: str, summary_text: str) -> float:
        source_words = len(source_text.split())
        summary_words = len(summary_text.split())
        return source_words / max(summary_words, 1)
    
    @staticmethod
    def calculate_repetition_score(text: str) -> float:
        if not text.strip():
            return 0.0
        
        words = text.lower().split()
        if len(words) < 4:
            return 0.0
        
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repeated = sum(count - 1 for count in bigram_counts.values() if count > 1)
        
        return repeated / max(len(bigrams), 1)
    
    @staticmethod
    def assess_document_complexity(text: str) -> dict:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        unique_words = len(set(word.lower().strip('.,!?;:"()[]') for word in words))
        lexical_diversity = unique_words / max(word_count, 1)
        
        length_complexity = min(word_count / 2000, 1.0)
        structure_complexity = min(avg_sentence_length / 35, 1.0)
        diversity_complexity = lexical_diversity
        
        overall_complexity = (length_complexity + structure_complexity + diversity_complexity) / 3
        
        if overall_complexity < 0.33:
            complexity_category = "simple"
        elif overall_complexity < 0.67:
            complexity_category = "medium"
        else:
            complexity_category = "complex"
        
        return {
            "word_count": word_count,
            "complexity_score": word_count,
            "complexity_category": complexity_category,
            "method": "length_based_percentile",
            "thresholds": {
                "simple_max": 416.0,
                "medium_max": 1539.0
            }
        }

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)

def run_real_mas_system(document: str, goal: str) -> dict:
    if not MAS_AVAILABLE:
        return {
            "summary": f"[MAS not available - would process {len(document)} chars]",
            "method": "real_mas",
            "error": "MAS system not available",
            "processing_time": 0,
            "tokens": 0
        }
    
    try:
        start_time = time.time()
        
        config = MASConfig(
            planner_config=AgentConfig(
                model="claude-3-5-sonnet-20241022", 
                temperature=0.1, 
                max_tokens=1500
            ),
            summarizer_config=AgentConfig(
                model="claude-3-5-sonnet-20241022", 
                temperature=0.3, 
                max_tokens=1000
            ),
            critic_config=AgentConfig(
                model="claude-3-5-sonnet-20241022", 
                temperature=0.2, 
                max_tokens=2000
            ),
            parallel_summarization=True,
            max_chunk_size=2000,
            enable_logging=False  # Disable for batch processing
        )
        
        mas_agent = MacroAgent(config)
        
        results = mas_agent.process_document(document, goal, save_intermediate=False)
        
        processing_time = time.time() - start_time
        
        final_summary = results["final_summary"]
        
        total_tokens = 0
        if "metadata" in results and "agent_stats" in results["metadata"]:
            agent_stats = results["metadata"]["agent_stats"]
            total_tokens += agent_stats.get("planner", {}).get("total_tokens", 0)
            total_tokens += agent_stats.get("summarizer", {}).get("total_tokens", 0)
            total_tokens += agent_stats.get("critic", {}).get("total_tokens", 0)
        
        if total_tokens == 0:
            total_tokens = estimate_tokens(document + final_summary)
        
        return {
            "summary": final_summary,
            "method": "real_mas",
            "processing_time": processing_time,
            "tokens": total_tokens,
            "mas_metadata": {
                "planner_output": results.get("planner_output", {}),
                "summarizer_output": results.get("summarizer_output", {}),
                "critic_output": results.get("critic_output", {}),
                "execution_stats": results.get("metadata", {})
            }
        }
        
    except Exception as e:
        return {
            "summary": f"Error in MAS processing: {str(e)}",
            "method": "real_mas",
            "error": str(e),
            "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
            "tokens": 0
        }

def process_dataset(dataset_name: str, start: int = 0, end: int = 75):
    
    datasets = {
        "xsum": {
            "file": "data/xsum_dev.jsonl",
            "doc_key": "document", 
            "sum_key": "summary",
            "goal": "Create a single sentence summary capturing the main point"
        },
        "cnn_dailymail": {
            "file": "data/cnn_dailymail_dev.jsonl",
            "doc_key": "article",
            "sum_key": "highlights", 
            "goal": "Create bullet-point highlights for news readers"
        },
        "govreport": {
            "file": "data/govreport_dev.jsonl",
            "doc_key": "report",
            "sum_key": "summary",
            "goal": "Create a comprehensive executive summary for policy makers"
        }
    }
    
    if dataset_name not in datasets:
        print(f"❌ Unknown dataset: {dataset_name}")
        return
    
    config = datasets[dataset_name]
    
    try:
        with open(config["file"], 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {config['file']}")
        return
    
    data = data[start:end]
    
    output_dir = Path("outputs/real_mas")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints" / dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🤖 Running REAL MAS SYSTEM on {dataset_name}")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    consecutive_failures = 0
    
    for i, row in enumerate(data, start=start):
        print(f"Processing {i+1-start}/{len(data)}: ", end="", flush=True)
        
        try:
            document = row[config["doc_key"]]
            reference = row[config["sum_key"]]
            goal = config["goal"]
            
            if isinstance(reference, list):
                reference = "\n".join(str(item) for item in reference)
            elif not isinstance(reference, str):
                reference = str(reference)
            
            complexity_analysis = AdvancedMetrics.assess_document_complexity(document)
            
            example_start_time = time.time()
            
            mas_result = run_real_mas_system(document, goal)
            
            example_processing_time = time.time() - example_start_time
            
            summary_text = mas_result["summary"]
            compression_ratio = AdvancedMetrics.calculate_compression_ratio(document, summary_text)
            repetition_score = AdvancedMetrics.calculate_repetition_score(summary_text)
            
            results.append({
                "id": i,
                "source": document,
                "reference": reference,
                "generated": summary_text,
                "method": "real_mas",
                "processing_time": example_processing_time,
                "complexity_analysis": complexity_analysis,
                "compression_ratio": compression_ratio,
                "repetition_score": repetition_score,
                "tokens": mas_result.get("tokens", 0),
                "mas_metadata": mas_result.get("mas_metadata", {})
            })
            
            consecutive_failures = 0
            
            tokens = mas_result.get('tokens', 0)
            print(f"✅ ({tokens} tokens)")
            
            if (i+1-start) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = len(data) - (i+1-start)
                eta = (elapsed / (i+1-start)) * remaining
                print(f"    📊 Progress: {i+1-start}/{len(data)} | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
            
            if (i+1-start) % 15 == 0:
                checkpoint_file = checkpoint_dir / f"checkpoint_{i+1-start:03d}.json"
                try:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"    💾 Checkpoint saved: {checkpoint_file} ({len(results)} results)")
                except Exception as e:
                    print(f"    ⚠️ Failed to save checkpoint: {e}")
                gc.collect()
            
            time.sleep(BASE_DELAY)
            
        except Exception as e:
            consecutive_failures += 1
            print(f"❌ Error: {e}")
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"🛑 Circuit breaker triggered: {consecutive_failures} consecutive failures")
                break
            
            results.append({
                "id": i,
                "source": document if 'document' in locals() else "",
                "reference": reference if 'reference' in locals() else "",
                "generated": f"Error: {str(e)}",
                "method": "real_mas",
                "error": str(e),
                "consecutive_failure_count": consecutive_failures
            })
    
    total_time = time.time() - start_time
    
    output_file = output_dir / f"real_mas_{dataset_name}_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        summary_file = output_dir / f"real_mas_{dataset_name}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Real MAS System Results - {dataset_name.upper()}\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Goal: {config['goal']}\n")
            f.write(f"Dataset file: {config['file']}\n")
            f.write(f"Total documents processed: {len(results)}\n")
            f.write(f"Total processing time: {total_time:.1f}s\n")
            f.write(f"Average time per document: {total_time/len(results):.1f}s\n")
            total_tokens = sum(r.get('tokens', 0) for r in results)
            f.write(f"Total tokens used: {total_tokens:,}\n")
            avg_compression = sum(r.get("compression_ratio", 0) for r in results) / len(results) if results else 0
            avg_repetition = sum(r.get("repetition_score", 0) for r in results) / len(results) if results else 0
            f.write(f"Average compression ratio: {avg_compression:.1f}x\n")
            f.write(f"Average repetition score: {avg_repetition:.3f}\n")
        
        print(f"\n📊 REAL MAS Results for {dataset_name}:")
        print(f"  ✅ Processed: {len(results)} examples")
        print(f"  ⏱️ Total time: {total_time:.1f}s ({total_time/len(results):.1f}s avg)")
        print(f"  🔤 Total tokens: {total_tokens:,}")
        print(f"  📏 Avg compression: {avg_compression:.1f}x")
        print(f"  🔄 Avg repetition: {avg_repetition:.3f}")
        print(f"  📁 Results: {output_file}")
        print(f"  📄 Summary: {summary_file}")
        print(f"  💾 Checkpoints: {checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ Failed to save final results: {e}")

def main():
    print("🚀 REAL MULTI-AGENT SYSTEM (MAS) RUNNER")
    print("=" * 60)
    
    if not MAS_AVAILABLE:
        print("❌ Real MAS system not available")
        print("Please ensure agents/macro_agent modules are properly installed")
        return
    
    print("✅ Real MAS system ready")
    print(f"Model: claude-3-5-sonnet-20241022")
    
    datasets = ["xsum", "cnn_dailymail", "govreport"]
    start_idx = 0
    end_idx = 75  # Match baseline
    
    print(f"\n⚙️ Configuration:")
    print(f"  Datasets: {datasets}")
    print(f"  Method: Real MAS with separate agent classes")
    print(f"  Examples: {start_idx} to {end_idx-1}")
    
    for dataset_name in datasets:
        try:
            process_dataset(dataset_name, start_idx, end_idx)
        except Exception as e:
            print(f"❌ Failed for {dataset_name}: {e}")
    
    print(f"\n🎉 REAL MAS PIPELINE COMPLETE!")
    print(f"📁 Check outputs in: outputs/real_mas/")
    print(f"📊 Results ready for comparison with baseline")

if __name__ == "__main__":
    main()