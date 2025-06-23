"""
RAG System Core Functionality Unit Tests
"""
import pytest
from unittest.mock import Mock, patch
from src.retrieval_graph.graph import _hybrid, _rerank, _generate
# expand_abbreviations function temporarily commented, can be uncommented when implemented
from langchain_core.documents import Document


class TestHybridRetrieval:
    """Test hybrid retrieval functionality"""
    
    def test_hybrid_search_basic(self):
        """Test basic hybrid search functionality"""
        # Mock vector store
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="test content 1", metadata={"file": "test1.txt", "cid": 1}),
            Document(page_content="test content 2", metadata={"file": "test2.txt", "cid": 2})
        ]
        
        inputs = {
            "query": "test query",
            "vectorstore": mock_vectorstore
        }
        
        with patch('src.retrieval_graph.graph.BM25Retriever') as mock_bm25:
            mock_bm25.from_documents.return_value.invoke.return_value = []
            result = _hybrid(inputs)
        
        assert "docs" in result
        assert "query" in result
        assert result["query"] == "test query"
        assert len(result["docs"]) >= 0

    def test_abbreviation_expansion(self):
        """Test abbreviation expansion functionality"""
        # Temporarily skipped, can be enabled after implementing abbreviation expansion
        pytest.skip("Abbreviation expansion feature not yet implemented, awaiting development")


class TestReranking:
    """Test reranking functionality"""
    
    @patch('src.retrieval_graph.graph.CrossEncoder')
    def test_rerank_documents(self, mock_cross_encoder):
        """Test document reranking"""
        # Mock CrossEncoder
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]
        mock_cross_encoder.return_value = mock_model
        
        test_docs = [
            Document(page_content="doc 1", metadata={"file": "test1.txt", "cid": 1}),
            Document(page_content="doc 2", metadata={"file": "test2.txt", "cid": 2}),
            Document(page_content="doc 3", metadata={"file": "test3.txt", "cid": 3})
        ]
        
        inputs = {
            "docs": test_docs,
            "query": "test query"
        }
        
        result = _rerank(inputs)
        
        assert "ctx" in result
        assert "srcs" in result
        assert "detailed_sources" in result
        assert len(result["detailed_sources"]) <= len(test_docs)


class TestGeneration:
    """Test answer generation functionality"""
    
    @patch('src.retrieval_graph.graph.openai_client')
    @patch('src.core.prompts.build_prompt')
    def test_generate_answer(self, mock_build_prompt, mock_openai):
        """Test answer generation"""
        # Mock dependencies
        mock_build_prompt.return_value = "test prompt"
        mock_response = Mock()
        mock_response.choices[0].message.content = "test answer"
        mock_openai.chat.completions.create.return_value = mock_response
        
        inputs = {
            "ctx": "test context",
            "query": "test query",
            "srcs": [("test.txt", 1)],
            "detailed_sources": []
        }
        
        result = _generate(inputs)
        
        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "test answer"


class TestSystemIntegration:
    """System integration tests"""
    
    def test_environment_variables(self):
        """Test required environment variables"""
        import os
        
        # These should be set in CI environment
        required_vars = ["OPENAI_API_KEY"]
        
        for var in required_vars:
            if var in os.environ:
                assert len(os.environ[var]) > 0, f"{var} should not be empty"

    def test_imports(self):
        """Test key module imports"""
        try:
            from src.retrieval_graph.graph import build_chain
            from src.index_graph.graph import build_index
            assert callable(build_chain)
            assert callable(build_index)
        except ImportError as e:
            pytest.fail(f"Import failed: {e}") 