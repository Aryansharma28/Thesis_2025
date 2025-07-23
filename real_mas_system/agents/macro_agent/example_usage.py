#!/usr/bin/env python3
"""Example usage of the Multi-Agent Summarization System."""

from macro_agent import MacroAgent
from utils.config import MASConfig, AgentConfig


def example_basic_usage():
    print("=== Basic Usage Example ===")
    
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
    
    goal = "Create a concise summary for a technology executive"
    
    agent = MacroAgent()
    
    results = agent.process_document(document, goal)
    
    print(f"Goal: {goal}")
    print(f"Original document length: {len(document)} characters")
    print(f"Final summary length: {len(results['final_summary'])} characters")
    print(f"Total execution time: {results['metadata']['total_execution_time']:.2f}s")
    print("\nFinal Summary:")
    print("-" * 40)
    print(results['final_summary'])
    print("\n")


def example_custom_config():
    print("=== Custom Configuration Example ===")
    
    custom_config = MASConfig(
        planner_config=AgentConfig(model="gpt-4", temperature=0.1, max_tokens=2000),
        summarizer_config=AgentConfig(model="gpt-4", temperature=0.2, max_tokens=1500),
        critic_config=AgentConfig(model="gpt-4", temperature=0.1, max_tokens=3000),
        parallel_summarization=False,
        max_chunk_size=1000,
        enable_logging=True,
        log_level="DEBUG"
    )
    
    document = """
    Climate change represents one of the most pressing challenges facing humanity today. The scientific consensus 
    is clear: human activities, particularly the emission of greenhouse gases, are driving unprecedented changes 
    in Earth's climate system.
    
    The primary driver of climate change is the increased concentration of carbon dioxide (CO2) in the atmosphere, 
    which has risen from approximately 280 parts per million (ppm) before the Industrial Revolution to over 410 ppm 
    today. This increase is primarily due to the burning of fossil fuels for energy production, transportation, 
    and industrial processes.
    
    The impacts of climate change are already visible worldwide. Rising global temperatures have led to melting 
    ice sheets and glaciers, contributing to sea-level rise. Extreme weather events, including hurricanes, droughts, 
    and heatwaves, are becoming more frequent and intense. These changes pose significant risks to ecosystems, 
    agriculture, water resources, and human health.
    
    Addressing climate change requires urgent action on multiple fronts. Mitigation efforts focus on reducing 
    greenhouse gas emissions through transitioning to renewable energy sources, improving energy efficiency, 
    and implementing carbon pricing mechanisms. Adaptation strategies involve preparing for and responding to 
    the unavoidable impacts of climate change, such as building resilient infrastructure and developing 
    climate-resistant crops.
    
    International cooperation is essential for effective climate action. The Paris Agreement, adopted in 2015, 
    represents a landmark global effort to limit warming to well below 2°C above pre-industrial levels. 
    However, current commitments are insufficient to meet this goal, highlighting the need for enhanced 
    ambition and implementation.
    """
    
    goal = "Prepare a comprehensive briefing for policy makers"
    
    agent = MacroAgent(custom_config)
    
    results = agent.process_document(document, goal, save_intermediate=True)
    
    print(f"Goal: {goal}")
    print(f"Configuration: Custom (sequential processing, smaller chunks)")
    print(f"Final summary:")
    print("-" * 40)
    print(results['final_summary'])
    print("\n")


def example_file_processing():
    print("=== File Processing Example ===")
    
    sample_content = """
    The Internet of Things (IoT) represents a network of interconnected devices that can collect and exchange data. 
    This technology has applications across numerous sectors including smart homes, industrial automation, 
    healthcare monitoring, and smart cities.
    
    In smart homes, IoT devices include thermostats, security cameras, lighting systems, and appliances that 
    can be controlled remotely. These devices learn user preferences and can automate routine tasks, improving 
    convenience and energy efficiency.
    
    Industrial IoT (IIoT) applications focus on manufacturing and supply chain optimization. Sensors monitor 
    equipment performance, predict maintenance needs, and track inventory levels. This leads to reduced 
    downtime, improved quality control, and cost savings.
    
    Healthcare IoT devices include wearable fitness trackers, remote patient monitoring systems, and smart 
    medical devices. These technologies enable continuous health monitoring, early detection of health issues, 
    and personalized treatment recommendations.
    
    Smart cities leverage IoT for traffic management, waste collection optimization, environmental monitoring, 
    and public safety. Connected infrastructure can reduce congestion, improve resource efficiency, and 
    enhance quality of life for residents.
    
    However, IoT adoption faces challenges including security vulnerabilities, privacy concerns, and 
    interoperability issues. As the number of connected devices grows exponentially, addressing these 
    challenges becomes increasingly critical for realizing IoT's full potential.
    """
    
    with open("sample_iot_document.txt", "w") as f:
        f.write(sample_content)
    
    agent = MacroAgent()
    goal = "Create a technical overview for software developers"
    
    try:
        final_summary = agent.process_file("sample_iot_document.txt", goal, "iot_summary.txt")
        
        print(f"Processed file: sample_iot_document.txt")
        print(f"Goal: {goal}")
        print(f"Output saved to: iot_summary.txt")
        print(f"Summary preview:")
        print("-" * 40)
        print(final_summary[:300] + "..." if len(final_summary) > 300 else final_summary)
        
    except FileNotFoundError:
        print("Sample file not found. Please ensure the file exists.")
    
    print("\n")


def example_error_handling():
    print("=== Error Handling Example ===")
    
    short_document = "AI is important."
    goal = "Create detailed analysis"
    
    agent = MacroAgent()
    
    try:
        results = agent.process_document(short_document, goal)
        print("Processed short document successfully")
        print(f"Summary: {results['final_summary']}")
        
        critic_output = results.get('critic_output', {})
        issues = critic_output.get('evaluation', {}).get('issues_found', {})
        if issues:
            print("Issues detected:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"- {issue_type}: {issue_list}")
    
    except Exception as e:
        print(f"Error processing document: {e}")
    
    print("\n")


def main():
    print("Multi-Agent Summarization System Examples")
    print("=" * 50)
    
    print("NOTE: These examples use placeholder API responses.")
    print("To use with real APIs, update the _make_api_call method in base_agent.py")
    print("=" * 50)
    print()
    
    example_basic_usage()
    example_custom_config()
    example_file_processing()
    example_error_handling()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()