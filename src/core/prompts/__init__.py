"""
Minimal prompt builder for CDC Weekly Reports RAG system
"""

def build_prompt(template_id: str, context: str, query: str) -> str:
    if template_id == "detailed":
        system_part = (
            "You are a CDC Weekly Reports assistant. Please provide answers strictly based on the provided content. "
            "If the information provided is insufficient to answer the question, respond with 'No relevant information available'."
        )
    else:
        system_part = "Please read the following context and answer the question."

    return (
        f"{system_part}\n\n"
        f"=== Context Information ===\n{context}\n\n"
        f"=== Question ===\n{query}"
    )
