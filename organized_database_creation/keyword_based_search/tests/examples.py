#!/usr/bin/env python3
"""
Example usage of Agriculture Data Curator
Demonstrates different configuration options and usage patterns
"""

import json
import time
from pathlib import Path
from agriculture_data_curator import AgricultureDataCurator, AgricultureSearchQueries

def example_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("Example 1: Basic Usage")
    print("=" * 50)
    
    curator = AgricultureDataCurator(
        num_agents=2,  # Fewer agents for testing
        model="deepseek-r1:70b",
        output_file="example_basic.jsonl",
        max_search_results=5
    )
    
    # Run with limited queries for testing
    summary = curator.start_curation(num_queries=10)
    
    print(f"Results: {summary}")
    return summary

def example_custom_queries():
    """Example 2: Using custom search queries"""
    print("\nExample 2: Custom Search Queries")
    print("=" * 50)
    
    # Define custom search queries focused on specific crops
    custom_queries = [
        "basmati rice cultivation Punjab India research data",
        "organic cotton farming Maharashtra sustainability reports",
        "millet cultivation drought resistance India studies",
        "sugarcane irrigation water management India",
        "spice export data India cardamom pepper statistics"
    ]
    
    curator = AgricultureDataCurator(
        num_agents=2,
        model="gemma3:27b",  # Alternative model
        output_file="example_custom.jsonl"
    )
    
    # Manually assign queries (you would modify the curator for this)
    print(f"Would process {len(custom_queries)} custom queries")
    return custom_queries

def example_analyze_output():
    """Example 3: Analyze output data"""
    print("\nExample 3: Analyzing Output Data")
    print("=" * 50)
    
    output_file = "example_basic.jsonl"
    
    if not Path(output_file).exists():
        print(f"Output file {output_file} not found. Run example_basic_usage() first.")
        return
    
    # Read and analyze the JSONL output
    entries = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Total entries: {len(entries)}")
    
    # Analyze by genre
    genres = {}
    for entry in entries:
        genre = entry.get('genre', 'unknown')
        genres[genre] = genres.get(genre, 0) + 1
    
    print("\nBy genre:")
    for genre, count in sorted(genres.items()):
        print(f"  {genre}: {count}")
    
    # Analyze by regions
    regions = {}
    for entry in entries:
        for region in entry.get('indian_regions', []):
            regions[region] = regions.get(region, 0) + 1
    
    print("\nTop regions mentioned:")
    for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {region}: {count}")
    
    # Analyze by crops
    crops = {}
    for entry in entries:
        for crop in entry.get('crop_types', []):
            crops[crop] = crops.get(crop, 0) + 1
    
    print("\nTop crops mentioned:")
    for crop, count in sorted(crops.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {crop}: {count}")
    
    # Average relevance score
    scores = [entry.get('relevance_score', 0) for entry in entries]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage relevance score: {avg_score:.3f}")
    
    return {
        'total_entries': len(entries),
        'genres': genres,
        'regions': regions,
        'crops': crops,
        'avg_relevance': avg_score
    }

def example_filtered_search():
    """Example 4: Filtered search for specific topics"""
    print("\nExample 4: Filtered Search")
    print("=" * 50)
    
    # Get queries for specific categories
    all_queries = AgricultureSearchQueries.get_search_queries()
    
    # Filter for specific topics
    organic_queries = [q for q in all_queries if 'organic' in q.lower()]
    tech_queries = [q for q in all_queries if any(term in q.lower() for term in ['precision', 'technology', 'digital'])]
    climate_queries = [q for q in all_queries if any(term in q.lower() for term in ['climate', 'drought', 'water'])]
    
    print(f"Found {len(organic_queries)} organic farming queries")
    print(f"Found {len(tech_queries)} technology queries") 
    print(f"Found {len(climate_queries)} climate-related queries")
    
    # You could run curator with these filtered queries
    return {
        'organic': organic_queries,
        'technology': tech_queries,
        'climate': climate_queries
    }

def example_performance_test():
    """Example 5: Performance testing with different configurations"""
    print("\nExample 5: Performance Testing")
    print("=" * 50)
    
    configs = [
        {"agents": 1, "results": 3, "queries": 5},
        {"agents": 2, "results": 5, "queries": 5},
        {"agents": 4, "results": 10, "queries": 5},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\nTesting config {i+1}: {config}")
        
        start_time = time.time()
        
        curator = AgricultureDataCurator(
            num_agents=config["agents"],
            model="deepseek-r1:70b",
            output_file=f"perf_test_{i+1}.jsonl",
            max_search_results=config["results"]
        )
        
        try:
            summary = curator.start_curation(num_queries=config["queries"])
            execution_time = time.time() - start_time
            
            result = {
                "config": config,
                "execution_time": execution_time,
                "total_entries": summary.get("total_entries", 0),
                "success": True
            }
            
        except Exception as e:
            result = {
                "config": config,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "success": False
            }
        
        results.append(result)
        print(f"Result: {result}")
    
    return results

def main():
    """Run all examples"""
    print("Agriculture Data Curator - Usage Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Queries", example_custom_queries),
        ("Analyze Output", example_analyze_output),
        ("Filtered Search", example_filtered_search),
        ("Performance Test", example_performance_test)
    ]
    
    for name, func in examples:
        try:
            print(f"\n\n{'='*20} {name} {'='*20}")
            result = func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print(f"\n\n{'='*60}")
    print("All examples completed!")
    print("Check the generated .jsonl files for results.")

if __name__ == "__main__":
    main()
