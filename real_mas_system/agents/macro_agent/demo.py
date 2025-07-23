#!/usr/bin/env python3
"""Demo script for the Multi-Agent Summarization System."""

import time
from adaptive_macro_agent import AdaptiveMacroAgent
from macro_agent import MacroAgent
from utils.config import MASConfig, AgentConfig
from utils.feedback_loop import FeedbackConfig


def demo_basic_summarization():
    print("🤖 BASIC MULTI-AGENT SUMMARIZATION DEMO")
    print("=" * 50)
    
    document = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
    From healthcare to transportation, AI is revolutionizing industries and changing how we live and work.
    
    In healthcare, AI applications include diagnostic imaging, drug discovery, and personalized treatment plans. 
    Machine learning algorithms can analyze medical images with accuracy that sometimes exceeds human specialists. 
    For instance, AI systems have shown remarkable success in detecting skin cancer from photographs and 
    identifying diabetic retinopathy from retinal scans.
    
    The transportation sector has seen significant advances with autonomous vehicles. Companies like Tesla, 
    Waymo, and Uber are developing self-driving cars that promise to reduce accidents and improve efficiency. 
    These vehicles use computer vision, sensor fusion, and deep learning to navigate complex environments.
    
    However, AI also presents challenges including job displacement, privacy concerns, and ethical considerations. 
    As AI systems become more sophisticated, ensuring they are fair, transparent, and aligned with human values 
    becomes increasingly important. Regulatory frameworks are being developed to address these concerns while 
    fostering innovation.
    
    Looking forward, AI is expected to continue its rapid development with advances in areas like natural language 
    processing, robotics, and quantum computing. The integration of AI into everyday life will likely accelerate, 
    making it essential for individuals and organizations to adapt to this technological shift.
    """
    
    goal = "Create a concise summary for business executives"
    
    print(f"📄 Document: {len(document)} characters")
    print(f"🎯 Goal: {goal}")
    print("\n🔄 Processing with Basic MacroAgent...")
    
    agent = MacroAgent()
    start_time = time.time()
    
    try:
        results = agent.process_document(document, goal)
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 Final summary length: {len(results['final_summary'])} characters")
        
        print(f"\n📝 FINAL SUMMARY:")
        print("-" * 30)
        print(results['final_summary'])
        
        metadata = results.get('metadata', {})
        agent_stats = metadata.get('agent_stats', {})
        
        print(f"\n📈 Agent Statistics:")
        for agent_name, stats in agent_stats.items():
            print(f"  {agent_name}: {stats.get('calls_made', 0)} API calls, {stats.get('execution_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Note: This demo uses placeholder API calls. To run with real APIs, update base_agent.py")
    
    print("\n" + "="*50 + "\n")


def demo_adaptive_summarization():
    print("🔄 ADAPTIVE MULTI-AGENT SUMMARIZATION DEMO")
    print("=" * 50)
    
    document = """
    Machine Learning Operations (MLOps) is a practice that aims to deploy and maintain machine learning models 
    in production reliably and efficiently. MLOps is a core function of Machine Learning Engineering, focused 
    on streamlining the process of taking machine learning models to production, and then maintaining and 
    monitoring them. The key components of MLOps include version control for data and models, automated testing 
    of ML models, continuous integration and continuous deployment (CI/CD) for ML systems, model monitoring 
    and alerting, and automated retraining pipelines.
    
    Data versioning ensures that the exact datasets used for training models are tracked and reproducible. 
    This is crucial for debugging model performance issues and understanding how different data affects model 
    outcomes. Tools like DVC (Data Version Control) and MLflow help manage data and model versioning.
    
    Model testing goes beyond traditional software testing to include tests for data quality, model performance, 
    and model behavior. This includes checking for data drift, concept drift, and ensuring that models meet 
    performance benchmarks before deployment.
    
    CI/CD for ML involves automated pipelines that can retrain models when new data becomes available, test 
    the updated models, and deploy them if they meet quality criteria. This automation reduces manual effort 
    and ensures consistent deployment processes.
    
    Monitoring in production involves tracking model performance metrics, data quality, and system health. 
    This helps detect when models are degrading and need to be retrained or when there are issues with the 
    data pipeline.
    
    However, MLOps faces several challenges including the complexity of ML workflows, the need for specialized 
    infrastructure, and the difficulty of debugging ML systems. Additionally, ensuring model fairness and 
    compliance with regulations adds another layer of complexity.
    """
    
    goal = "Create a technical summary for software engineers new to MLOps"
    
    feedback_config = FeedbackConfig(
        min_confidence_threshold=0.7,
        min_individual_score_threshold=0.6,
        max_feedback_iterations=2,
        enable_final_critic_pass=True
    )
    
    mas_config = MASConfig(
        parallel_summarization=True,
        max_chunk_size=1500,
        enable_adaptive_feedback=True,
        enable_logging=True
    )
    
    print(f"📄 Document: {len(document)} characters")
    print(f"🎯 Goal: {goal}")
    print(f"⚙️  Config: {feedback_config.max_feedback_iterations} max iterations, {feedback_config.min_confidence_threshold} confidence threshold")
    print("\n🔄 Processing with Adaptive MacroAgent...")
    
    agent = AdaptiveMacroAgent(mas_config, feedback_config)
    start_time = time.time()
    
    try:
        results = agent.process_document_adaptive(document, goal, save_intermediate=True)
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing completed in {processing_time:.2f}s")
        
        feedback_results = results['feedback_loop_results']
        improvement_metrics = results['adaptive_metadata']['improvement_achieved']
        
        print(f"\n🔄 Feedback Loop Results:")
        print(f"  Iterations performed: {feedback_results['total_iterations']}")
        print(f"  Summaries improved: {improvement_metrics['summaries_improved']}")
        print(f"  Confidence improvement: {improvement_metrics['confidence_improvement']:.3f}")
        print(f"  Issues reduced: {improvement_metrics['issues_reduced']}")
        
        if feedback_results['feedback_iterations']:
            print(f"\n📊 Iteration Details:")
            for i, iteration in enumerate(feedback_results['feedback_iterations'], 1):
                print(f"  Iteration {i}: {iteration['summaries_retried']} summaries retried")
        
        print(f"\n📝 FINAL SUMMARY:")
        print("-" * 30)
        print(results['final_summary'])
        
        print(f"\n📈 Quality Metrics:")
        final_quality = results['adaptive_metadata']['final_quality_scores']
        print(f"  Final quality score: {final_quality:.3f}")
        print(f"  Total execution time: {processing_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Note: This demo uses placeholder API calls. To run with real APIs, update base_agent.py")
    
    print("\n" + "="*50 + "\n")


def show_system_architecture():
    print("🏗️  SYSTEM ARCHITECTURE")
    print("=" * 50)
    
    print("""
📋 BASIC FLOW:
Document → Planner → Summarizer(s) → Critic → Final Summary

🔄 ADAPTIVE FLOW:
Document → Planner → Summarizer(s) → Critic → Quality Check
                                                    ↓ (if low quality)
                                              Retry Tasks ← Feedback Analysis
                                                    ↓
                                        Enhanced Summarizer(s) → Updated Results
                                                    ↓
                                              Final Critic → Final Summary

🤖 AGENTS:
• Planner Agent: Analyzes document structure and creates optimal chunking strategy
• Summarizer Agent: Processes chunks in parallel with specific instructions
• Critic Agent: Evaluates summaries and synthesizes final output
• Enhanced Critic: Adds detailed quality assessment for feedback loops

⚙️  FEEDBACK SYSTEM:
• Quality Detection: Identifies summaries below thresholds
• Issue Analysis: Detects specific problems (confidence, coherence, etc.)
• Adaptive Instructions: Creates improved prompts for retry
• Iterative Improvement: Continues until quality targets are met

📊 CONFIGURATION:
• Quality thresholds (confidence, individual scores)
• Maximum feedback iterations
• Temperature adjustments for retries
• Critical issue detection patterns
    """)
    
    print("=" * 50 + "\n")


def main():
    print("🎉 MULTI-AGENT SUMMARIZATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases both basic and adaptive summarization capabilities.")
    print("Note: Uses placeholder API responses - implement real API calls in base_agent.py")
    print("=" * 60 + "\n")
    
    show_system_architecture()
    
    demo_basic_summarization()
    
    demo_adaptive_summarization()
    
    print("🎯 SUMMARY:")
    print("- Basic MAS provides structured multi-agent summarization")
    print("- Adaptive MAS adds feedback loops for quality improvement") 
    print("- System is modular and configurable for different use cases")
    print("- Ready for integration with OpenAI/Anthropic APIs")
    print("\n✨ Demo completed!")


if __name__ == "__main__":
    main()