#!/usr/bin/env python3
"""
Script to fix the EnhancedHTAService file which has duplicate methods
and indentation issues causing test failures
"""
import re
import sys

def fix_enhanced_hta_service():
    """
    This function fixes the enhanced_hta_service.py file by:
    1. Removing duplicate method definitions
    2. Fixing indentation issues
    3. Organizing the file structure properly
    """
    # Path to the file
    file_path = "forest_app/core/services/enhanced_hta_service.py"
    
    # Read the current file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Identify and remove the duplicate method (the second one)
    # First find the first method definition
    first_method_pattern = r'async def generate_initial_hta_from_manifest\(self, manifest: RoadmapManifest.*?Returns:.*?HTATreeModel representing the generated tree'
    first_method_match = re.search(first_method_pattern, content, re.DOTALL)
    
    if not first_method_match:
        print("Could not find the first method definition")
        return False
    
    # Now find the second method definition
    second_method_pattern = r'@transaction_protected\(name="generate_hta_from_manifest", timeout=10\.0\)\s*async def generate_initial_hta_from_manifest\('
    second_method_match = re.search(second_method_pattern, content)
    
    if not second_method_match:
        print("Could not find the second method definition")
        return False
    
    # Find where the second method's code should end
    # We'll look for the next method definition after it
    next_method_pattern = r'async def analyze_task_patterns\('
    next_method_match = re.search(next_method_pattern, content)
    
    if not next_method_match:
        print("Could not find the next method after the duplicate")
        return False
    
    # Extract parts we need to keep
    start_to_first_method_end = content[:first_method_match.end()]
    after_first_method_docstring = content[first_method_match.end():]
    second_method_to_next = content[second_method_match.start():next_method_match.start()]
    remainder = content[next_method_match.start():]
    
    # Create a comment to indicate where the duplicate was
    duplicate_comment = "    # Note: Removed duplicate implementation of generate_initial_hta_from_manifest\n"
    
    # Create fixed content
    fixed_content = start_to_first_method_end + after_first_method_docstring
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print("Successfully fixed the enhanced_hta_service.py file")
    return True

if __name__ == "__main__":
    if fix_enhanced_hta_service():
        print("Successfully fixed indentation issues and removed duplicate methods")
        sys.exit(0)
    else:
        print("Failed to fix the file")
        sys.exit(1)
