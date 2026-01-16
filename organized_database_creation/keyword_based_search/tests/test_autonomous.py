#!/usr/bin/env python3
"""
Quick test of autonomous agriculture curator with 3 agents
"""

import logging
from autonomous_agriculture_curator import AutonomousAgricultureCurator

def main():
    """Test autonomous curator with limited agents and searches"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 Testing Autonomous Agriculture Curator")
    print("=" * 50)
    
    # Test with 3 agents, 10 searches each
    curator = AutonomousAgricultureCurator(
        num_agents=3,
        output_file="test_autonomous_agriculture.jsonl",
        max_search_results=10,
        pdf_storage_dir="test_autonomous_pdfs",
        enable_pdf_download=True,
        searches_per_agent=10  # Small test
    )
    
    try:
        summary = curator.start_autonomous_curation()
        
        print("\n" + "="*60)
        print("🎉 AUTONOMOUS TEST COMPLETED")
        print("="*60)
        print(f"✅ Total entries: {summary.get('total_entries', 0)}")
        print(f"📝 Written to JSONL: {summary.get('actual_entries_written', 0)}")
        print(f"🔍 Autonomous searches: {summary.get('total_autonomous_searches', 0)}")
        print(f"🌐 Unique URLs: {summary.get('total_unique_urls', 0)}")
        print(f"⏱️ Time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"🤖 Agents: {summary.get('successful_agents', 0)}/{summary.get('agents_deployed', 0)}")
        
        # Show sample of agent performance
        print("\n📊 Agent Performance:")
        for agent_perf in summary.get('agent_performance', []):
            print(f"  Agent {agent_perf['agent_id']}: {agent_perf['specialization']}")
            print(f"    📝 {agent_perf['entries_collected']} entries")
            print(f"    🔍 {agent_perf['autonomous_searches']} searches")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
