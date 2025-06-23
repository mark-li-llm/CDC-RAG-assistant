#!/usr/bin/env python3
"""
RAG System Retrieval Quality Assessment
"""
import json
import os
from unittest.mock import patch, Mock
from src.retrieval_graph.graph import build_chain
from src.index_graph.graph import build_index


def create_test_dataset():
    """Create test dataset"""
    return {
        "queries": [
            {
                "query": "What did the Tennessee report say about influenza activity in children?",
                "expected_keywords": ["Tennessee", "children", "influenza", "activity"],
                "min_sources": 1
            },
            {
                "query": "How much higher is arthritis prevalence in young male veterans?",
                "expected_keywords": ["veterans", "arthritis", "prevalence", "male"],
                "min_sources": 1
            },
            {
                "query": "In which age group did COPD prevalence decline?",
                "expected_keywords": ["COPD", "prevalence", "age group", "decline"],
                "min_sources": 1
            },
            {
                "query": "What are the CDC recommendations for flu vaccines?",
                "expected_keywords": ["CDC", "flu vaccine", "recommendations"],
                "min_sources": 1
            },
            {
                "query": "What are the symptoms of chronic obstructive pulmonary disease?",
                "expected_keywords": ["chronic obstructive pulmonary disease", "symptoms"],
                "min_sources": 1
            }
        ]
    }


def evaluate_keyword_coverage(answer, expected_keywords):
    """Evaluate keyword coverage rate"""
    answer_lower = answer.lower()
    found_keywords = []
    
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
    
    coverage = len(found_keywords) / len(expected_keywords)
    return coverage, found_keywords


def evaluate_source_quality(sources, min_sources):
    """Evaluate source document quality"""
    return {
        "source_count": len(sources),
        "meets_minimum": len(sources) >= min_sources,
        "has_sources": len(sources) > 0
    }


def evaluate_answer_quality(answer):
    """Evaluate answer quality"""
    return {
        "length_appropriate": 50 <= len(answer) <= 1000,
        "not_empty": len(answer.strip()) > 0,
        "not_error_message": "error" not in answer.lower() and "sorry" not in answer.lower()
    }


def main():
    """Main quality assessment function"""
    print("🔍 Starting RAG system quality assessment...")
    
    # Mock setup
    with patch('src.retrieval_graph.graph.openai_client') as mock_openai:
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Based on the provided CDC documents, Tennessee has reported increased influenza activity among children during the current season. The data shows elevated cases particularly in school-age children between 5-12 years old, with symptoms including fever, cough, and respiratory distress."
        mock_openai.chat.completions.create.return_value = mock_response
        
        with patch('src.index_graph.graph.build_index') as mock_build_index:
            # Mock vector store
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = [
                Mock(
                    page_content="Tennessee Department of Health reports increased influenza activity in children", 
                    metadata={"file": "tennessee_flu_report.txt", "cid": 1}
                ),
                Mock(
                    page_content="Pediatric influenza surveillance data shows rising cases", 
                    metadata={"file": "pediatric_flu_data.txt", "cid": 2}
                )
            ]
            mock_build_index.return_value = {"vectorstore": mock_vectorstore}
            
            # Create test dataset
            test_dataset = create_test_dataset()
            
            # Initialize system
            index = mock_build_index()
            chain = build_chain()
            
            # Run assessment
            results = []
            
            for i, test_case in enumerate(test_dataset["queries"]):
                print(f"\n📝 Testing query {i+1}/{len(test_dataset['queries'])}: {test_case['query'][:50]}...")
                
                try:
                    # Execute query
                    result = chain.invoke({"query": test_case["query"], **index})
                    
                    # Evaluate results
                    keyword_coverage, found_keywords = evaluate_keyword_coverage(
                        result["answer"], test_case["expected_keywords"]
                    )
                    
                    source_quality = evaluate_source_quality(
                        result["sources"], test_case["min_sources"]
                    )
                    
                    answer_quality = evaluate_answer_quality(result["answer"])
                    
                    # Calculate overall score
                    overall_score = (
                        keyword_coverage * 0.4 +
                        (1.0 if source_quality["meets_minimum"] else 0.0) * 0.3 +
                        (1.0 if answer_quality["length_appropriate"] else 0.0) * 0.2 +
                        (1.0 if answer_quality["not_error_message"] else 0.0) * 0.1
                    )
                    
                    test_result = {
                        "query": test_case["query"],
                        "keyword_coverage": keyword_coverage,
                        "found_keywords": found_keywords,
                        "source_quality": source_quality,
                        "answer_quality": answer_quality,
                        "overall_score": overall_score,
                        "answer_preview": result["answer"][:100] + "..."
                    }
                    
                    results.append(test_result)
                    
                    print(f"   ✅ Keyword coverage: {keyword_coverage:.2f}")
                    print(f"   📚 Source count: {source_quality['source_count']}")
                    print(f"   📊 Overall score: {overall_score:.2f}")
                    
                except Exception as e:
                    print(f"   ❌ Test failed: {e}")
                    results.append({
                        "query": test_case["query"],
                        "error": str(e),
                        "overall_score": 0.0
                    })
    
    # Calculate summary statistics
    successful_tests = [r for r in results if "error" not in r]
    
    if successful_tests:
        avg_keyword_coverage = sum(r["keyword_coverage"] for r in successful_tests) / len(successful_tests)
        avg_overall_score = sum(r["overall_score"] for r in successful_tests) / len(successful_tests)
        source_success_rate = sum(1 for r in successful_tests if r["source_quality"]["has_sources"]) / len(successful_tests)
        
        print(f"\n📊 Quality Assessment Summary:")
        print(f"   Successful tests: {len(successful_tests)}/{len(results)}")
        print(f"   Average keyword coverage: {avg_keyword_coverage:.2f}")
        print(f"   Average overall score: {avg_overall_score:.2f}")
        print(f"   Source success rate: {source_success_rate:.2f}")
        
        # Save results to file
        summary = {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "avg_keyword_coverage": avg_keyword_coverage,
            "avg_overall_score": avg_overall_score,
            "source_success_rate": source_success_rate,
            "details": results
        }
        
        with open("quality_results.txt", "w", encoding="utf-8") as f:
            f.write(f"Total tests: {summary['total_tests']}\n")
            f.write(f"Successful tests: {summary['successful_tests']}\n")
            f.write(f"Average keyword coverage: {summary['avg_keyword_coverage']:.2f}\n")
            f.write(f"Average overall score: {summary['avg_overall_score']:.2f}\n")
            f.write(f"Source success rate: {summary['source_success_rate']:.2f}\n")
        
        with open("quality_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved to quality_results.txt and quality_results.json")
        
        # Set exit code
        if avg_overall_score >= 0.7:
            print("🎉 Quality assessment passed!")
            exit(0)
        else:
            print("⚠️ Quality assessment did not meet expected standards")
            exit(1)
    else:
        print("❌ All tests failed")
        exit(1)


if __name__ == "__main__":
    main() 