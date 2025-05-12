"""
Policy dataset utility functions for loading acceptance and rejection policy datasets.
"""
import re
import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _find_data_directory() -> str:
    """
    Helper function to locate the data directory.
    
    Returns:
        Path to the data directory
        
    Raises:
        FileNotFoundError: If no data directory can be found
    """
    possible_locations = ['data', '../data', '../../data', '.']
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
    
    raise FileNotFoundError("Could not find data directory")


def load_acceptance_policies(base_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load the acceptance policies dataset.
    
    Args:
        base_dir: Optional base directory containing the data folders.
                 If None, tries to automatically find the data directory.
    
    Returns:
        List of policy dictionaries from the acceptance dataset
    
    Raises:
        FileNotFoundError: If the dataset file cannot be found
    """
    # Determine the base directory
    if base_dir is None:
        base_dir = _find_data_directory()
    
    # Build the file path
    file_path = os.path.join(base_dir, "acceptance_policies", "acceptance_policies_dataset.json")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Acceptance policies dataset not found at {file_path}")


def load_rejection_policies(base_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load the rejection policies dataset.
    
    Args:
        base_dir: Optional base directory containing the data folders.
                 If None, tries to automatically find the data directory.
    
    Returns:
        List of policy dictionaries from the rejection dataset
    
    Raises:
        FileNotFoundError: If the dataset file cannot be found
    """
    # Determine the base directory
    if base_dir is None:
        base_dir = _find_data_directory()
    
    # Build the file path
    file_path = os.path.join(base_dir, "rejection_policies", "rejection_policies_dataset.json")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rejection policies dataset not found at {file_path}")


# Original function for backward compatibility
def load_policy_dataset(dataset_type: str, base_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load a policy dataset from the appropriate JSON file.
    
    Args:
        dataset_type: Either 'rejection' or 'acceptance' to specify which dataset to load
        base_dir: Base directory containing the data folders.
    
    Returns:
        List of policy dictionaries from the dataset
    
    Raises:
        ValueError: If dataset_type is not 'rejection' or 'acceptance'
        FileNotFoundError: If the dataset file cannot be found
    """
    if dataset_type not in ['rejection', 'acceptance']:
        raise ValueError("dataset_type must be either 'rejection' or 'acceptance'")
    
    if dataset_type == 'rejection':
        return load_rejection_policies(base_dir)
    else:
        return load_acceptance_policies(base_dir)
    
def extract_confidence_score(text: str) -> float:
    """Extract a confidence score from the LLM response"""
    try:
        # Look for patterns like "confidence score: 85" or "confidence: 85/100"
        patterns = [
            r'confidence score[^\d]*(\d+)',
            r'confidence[^\d]*(\d+)',
            r'score[^\d]*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                score = float(matches[0])
                # Normalize to 0-1 range if it's on a 0-100 scale
                return score / 100 if score > 1 else score
        
        # Default score if no match is found
        return 0.7
    except Exception as e:
        logger.warning(f"Error extracting confidence score: {e}")
        return 0.7  # Default moderate confidence


if __name__ == "__main__":
    # Example usage
    try:
        # Load acceptance policies
        acceptance_policies = load_acceptance_policies()
        print(f"Loaded {len(acceptance_policies)} acceptance policies")
        
        # Load rejection policies
        rejection_policies = load_rejection_policies()
        print(f"Loaded {len(rejection_policies)} rejection policies")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")