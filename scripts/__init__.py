"""
PaperScout - Retrieval and Data Processing Package.

This package exposes the core retrieval interface for downstream LLM and Streamlit integration.
"""

from .retrieve_papers import retrieve_papers

__all__ = ["retrieve_papers"]
