"""
RAG System Performance Benchmark Tests
"""
import pytest
import time
from unittest.mock import Mock, patch
from src.retrieval_graph.graph import build_chain
from src.index_graph.graph import build_index


class TestRAGPerformance:
    """RAG system performance tests"""
    
    def setup_method(self):
        """Test setup"""
        # Mock external dependencies to avoid API calls
        self.mock_openai_patcher = patch('src.retrieval_graph.graph.openai_client')
        self.mock_openai = self.mock_openai_patcher.start()
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test answer response"
        self.mock_openai.chat.completions.create.return_value = mock_response

    def teardown_method(self):
        """Test cleanup"""
        self.mock_openai_patcher.stop()

    @pytest.mark.benchmark
    def test_single_query_performance(self, benchmark):
        """Test single query performance"""
        
        def run_single_query():
            # Create test environment
            with patch('src.index_graph.graph.build_index') as mock_build_index:
                # Mock vector store
                mock_vectorstore = Mock()
                mock_vectorstore.similarity_search.return_value = [
                    Mock(page_content="test content", metadata={"file": "test.txt", "cid": 1})
                ]
                mock_build_index.return_value = {"vectorstore": mock_vectorstore}
                
                # Execute query
                index = mock_build_index()
                chain = build_chain()
                result = chain.invoke({"query": "test query", **index})
                return result
        
        # Benchmark test
        result = benchmark(run_single_query)
        assert "answer" in result
        assert "sources" in result

    @pytest.mark.benchmark
    def test_multiple_queries_performance(self, benchmark):
        """Test multiple queries performance"""
        
        test_queries = [
            "What is COPD?",
            "Tennessee children flu activity report",
            "CDC latest recommendations",
            "Arthritis in veterans",
            "Public health guidance"
        ]
        
        def run_multiple_queries():
            results = []
            with patch('src.index_graph.graph.build_index') as mock_build_index:
                mock_vectorstore = Mock()
                mock_vectorstore.similarity_search.return_value = [
                    Mock(page_content="test content", metadata={"file": "test.txt", "cid": 1})
                ]
                mock_build_index.return_value = {"vectorstore": mock_vectorstore}
                
                index = mock_build_index()
                chain = build_chain()
                
                for query in test_queries:
                    result = chain.invoke({"query": query, **index})
                    results.append(result)
            
            return results
        
        results = benchmark(run_multiple_queries)
        assert len(results) == len(test_queries)
        for result in results:
            assert "answer" in result

    def test_response_time_under_threshold(self):
        """Ensure response time is within acceptable range"""
        
        with patch('src.index_graph.graph.build_index') as mock_build_index:
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = [
                Mock(page_content="test content", metadata={"file": "test.txt", "cid": 1})
            ]
            mock_build_index.return_value = {"vectorstore": mock_vectorstore}
            
            index = mock_build_index()
            chain = build_chain()
            
            start_time = time.time()
            result = chain.invoke({"query": "test query", **index})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Ensure response time is less than 5 seconds
            assert response_time < 5.0, f"Response time {response_time:.2f}s exceeds threshold"
            assert "answer" in result

    @pytest.mark.benchmark(group="retrieval")
    def test_retrieval_performance(self, benchmark):
        """Test retrieval component performance"""
        
        def run_retrieval():
            from src.retrieval_graph.graph import _hybrid
            
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = [
                Mock(page_content=f"content {i}", metadata={"file": f"test{i}.txt", "cid": i})
                for i in range(10)
            ]
            
            with patch('src.retrieval_graph.graph.BM25Retriever') as mock_bm25:
                mock_bm25.from_documents.return_value.invoke.return_value = []
                
                inputs = {
                    "query": "test query",
                    "vectorstore": mock_vectorstore
                }
                
                result = _hybrid(inputs)
                return result
        
        result = benchmark(run_retrieval)
        assert "docs" in result
        assert "query" in result

    @pytest.mark.benchmark(group="reranking")
    def test_reranking_performance(self, benchmark):
        """Test reranking component performance"""
        
        def run_reranking():
            from src.retrieval_graph.graph import _rerank
            
            test_docs = [
                Mock(page_content=f"document {i}", metadata={"file": f"test{i}.txt", "cid": i})
                for i in range(15)  # Test reranking of 15 documents
            ]
            
            with patch('src.retrieval_graph.graph.CrossEncoder') as mock_cross_encoder:
                mock_model = Mock()
                mock_model.predict.return_value = [0.9 - i*0.05 for i in range(15)]
                mock_cross_encoder.return_value = mock_model
                
                inputs = {
                    "docs": test_docs,
                    "query": "test query"
                }
                
                result = _rerank(inputs)
                return result
        
        result = benchmark(run_reranking)
        assert "ctx" in result
        assert "srcs" in result


class TestMemoryUsage:
    """Memory usage tests"""
    
    def test_memory_efficient_processing(self):
        """Test memory efficiency"""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Record initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate multiple queries
        with patch('src.index_graph.graph.build_index') as mock_build_index:
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = [
                Mock(page_content="test content", metadata={"file": "test.txt", "cid": 1})
            ]
            mock_build_index.return_value = {"vectorstore": mock_vectorstore}
            
            # Execute multiple queries
            index = mock_build_index()
            chain = build_chain()
            
            for i in range(50):
                result = chain.invoke({"query": f"test query {i}", **index})
                assert "answer" in result
        
        # Record final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Ensure memory increase is within reasonable range (< 100MB)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB" 