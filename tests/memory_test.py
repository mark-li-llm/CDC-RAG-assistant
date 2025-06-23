#!/usr/bin/env python3
"""
RAG System Memory Usage Monitoring
"""
import time
import os
import psutil
from typing import Dict, List
from unittest.mock import patch, Mock


class MemoryMonitor:
    """System memory monitoring utility"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements: List[Dict] = []
    
    def start_monitoring(self):
        """Start memory monitoring"""
        initial_memory = self.process.memory_info()
        self.measurements.append({
            "timestamp": time.time(),
            "rss_mb": initial_memory.rss / 1024 / 1024,
            "vms_mb": initial_memory.vms / 1024 / 1024,
            "event": "monitoring_started"
        })
        print(f"📊 Memory monitoring started. Initial RSS: {initial_memory.rss / 1024 / 1024:.2f} MB")
    
    def record_measurement(self, event: str):
        """Record memory measurement"""
        current_memory = self.process.memory_info()
        self.measurements.append({
            "timestamp": time.time(),
            "rss_mb": current_memory.rss / 1024 / 1024,
            "vms_mb": current_memory.vms / 1024 / 1024,
            "event": event
        })
        print(f"   📈 {event}: RSS {current_memory.rss / 1024 / 1024:.2f} MB")
    
    def get_memory_increase(self) -> float:
        """Calculate total memory increase"""
        if len(self.measurements) < 2:
            return 0.0
        
        initial = self.measurements[0]["rss_mb"]
        final = self.measurements[-1]["rss_mb"]
        return final - initial
    
    def get_max_memory(self) -> float:
        """Get maximum memory usage"""
        if not self.measurements:
            return 0.0
        return max(m["rss_mb"] for m in self.measurements)
    
    def save_report(self, filename: str = "memory_report.txt"):
        """Save memory usage report"""
        with open(filename, "w") as f:
            f.write("RAG System Memory Usage Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total measurements: {len(self.measurements)}\n")
            f.write(f"Initial memory: {self.measurements[0]['rss_mb']:.2f} MB\n")
            f.write(f"Final memory: {self.measurements[-1]['rss_mb']:.2f} MB\n")
            f.write(f"Memory increase: {self.get_memory_increase():.2f} MB\n")
            f.write(f"Peak memory: {self.get_max_memory():.2f} MB\n\n")
            
            f.write("Detailed measurements:\n")
            f.write("-" * 30 + "\n")
            for i, measurement in enumerate(self.measurements):
                f.write(f"{i+1:2d}. {measurement['event']:25s} - RSS: {measurement['rss_mb']:6.2f} MB\n")


def run_memory_stress_test():
    """Run memory stress test"""
    print("🧪 Starting RAG system memory stress test...")
    
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    
    # Mock setup to avoid real API calls
    with patch('src.retrieval_graph.graph.openai_client') as mock_openai:
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test answer for memory monitoring"
        mock_openai.chat.completions.create.return_value = mock_response
        
        with patch('src.index_graph.graph.build_index') as mock_build_index:
            # Mock vector store
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = [
                Mock(page_content="test content", metadata={"file": "test.txt", "cid": 1})
            ]
            mock_build_index.return_value = {"vectorstore": mock_vectorstore}
            
            monitor.record_measurement("system_initialization")
            
            # Import after mock setup
            from src.retrieval_graph.graph import build_chain
            
            # Initialize system
            index = mock_build_index()
            chain = build_chain()
            monitor.record_measurement("chain_built")
            
            # Test different query scenarios
            test_scenarios = [
                ("Short query test", ["What is COPD?"]),
                ("Multiple queries", [f"Query {i}" for i in range(10)]),
                ("Long query test", ["What are the detailed symptoms, causes, and treatment options for chronic obstructive pulmonary disease in elderly patients with comorbidities?"]),
                ("Batch processing", [f"Complex medical query about topic {i}" for i in range(25)])
            ]
            
            for scenario_name, queries in test_scenarios:
                print(f"\n🔬 Running scenario: {scenario_name}")
                
                for i, query in enumerate(queries):
                    try:
                        result = chain.invoke({"query": query, **index})
                        
                        # Verify result structure
                        assert "answer" in result
                        assert "sources" in result
                        
                        # Record measurement every 5 queries to track memory growth
                        if (i + 1) % 5 == 0:
                            monitor.record_measurement(f"{scenario_name}_query_{i+1}")
                    
                    except Exception as e:
                        print(f"   ❌ Query failed: {e}")
                        continue
                
                monitor.record_measurement(f"{scenario_name}_completed")
            
            # Cleanup test
            print("\n🧹 Testing cleanup and garbage collection...")
            import gc
            gc.collect()
            monitor.record_measurement("garbage_collection")
    
    # Final analysis
    print("\n📋 Memory test analysis:")
    memory_increase = monitor.get_memory_increase()
    max_memory = monitor.get_max_memory()
    
    print(f"   Total memory increase: {memory_increase:.2f} MB")
    print(f"   Peak memory usage: {max_memory:.2f} MB")
    
    # Save report
    monitor.save_report()
    print(f"✅ Memory report saved to memory_report.txt")
    
    # Evaluation
    MEMORY_INCREASE_THRESHOLD = 100  # MB
    PEAK_MEMORY_THRESHOLD = 500     # MB
    
    if memory_increase > MEMORY_INCREASE_THRESHOLD:
        print(f"⚠️ WARNING: Memory increase ({memory_increase:.2f} MB) exceeds threshold ({MEMORY_INCREASE_THRESHOLD} MB)")
        return False
    
    if max_memory > PEAK_MEMORY_THRESHOLD:
        print(f"⚠️ WARNING: Peak memory ({max_memory:.2f} MB) exceeds threshold ({PEAK_MEMORY_THRESHOLD} MB)")
        return False
    
    print("✅ Memory test passed!")
    return True


def check_memory_leaks():
    """Check for potential memory leaks"""
    print("\n🔍 Checking for memory leaks...")
    
    import gc
    
    # Get initial object count
    gc.collect()
    initial_objects = len(gc.get_objects())
    
    # Run multiple operations
    with patch('src.retrieval_graph.graph.openai_client') as mock_openai:
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        with patch('src.index_graph.graph.build_index') as mock_build_index:
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = [
                Mock(page_content="test", metadata={"file": "test.txt", "cid": 1})
            ]
            mock_build_index.return_value = {"vectorstore": mock_vectorstore}
            
            from src.retrieval_graph.graph import build_chain
            
            # Run multiple operations
            for i in range(10):
                index = mock_build_index()
                chain = build_chain()
                result = chain.invoke({"query": f"test query {i}", **index})
                
                # Force cleanup
                del index, chain, result
                gc.collect()
    
    # Get final object count
    gc.collect()
    final_objects = len(gc.get_objects())
    
    object_increase = final_objects - initial_objects
    print(f"   Object count increase: {object_increase}")
    
    # Allow some object growth but flag significant increases
    if object_increase > 1000:
        print(f"⚠️ WARNING: Significant object count increase detected: {object_increase}")
        return False
    
    print("✅ No significant memory leaks detected")
    return True


def main():
    """Main memory test function"""
    print("🚀 Starting comprehensive memory analysis...\n")
    
    # Run stress test
    stress_test_passed = run_memory_stress_test()
    
    # Check for memory leaks
    leak_test_passed = check_memory_leaks()
    
    # Final summary
    print(f"\n📊 Memory Analysis Summary:")
    print(f"   Stress test: {'✅ PASSED' if stress_test_passed else '❌ FAILED'}")
    print(f"   Leak test: {'✅ PASSED' if leak_test_passed else '❌ FAILED'}")
    
    if stress_test_passed and leak_test_passed:
        print("\n🎉 All memory tests passed!")
        return True
    else:
        print("\n⚠️ Memory tests failed - review results and optimize")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 