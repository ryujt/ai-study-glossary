import re
import os
import argparse
import sys
from pathlib import Path

def sanitize_filename(term):
    """Convert term to a valid filename (same as in create-explanations.py)."""
    # Remove characters that aren't allowed in filenames
    sanitized = re.sub(r'[\\/*?:"<>|]', '', term)
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[\s\-–]', '_', sanitized)
    # Remove any other problematic characters
    sanitized = re.sub(r'[^\w\._]', '', sanitized)
    return sanitized

def add_explanation_links(update_original=False):
    """Add links to explanation files in README.md."""
    # Check if README.md exists
    if not os.path.exists("README.md"):
        print("Error: README.md file not found in the current directory.")
        return False
    
    # Check if explanations directory exists
    explanations_dir = "explanations"
    if not os.path.isdir(explanations_dir):
        print(f"Error: {explanations_dir} directory not found. Please run create-explanations.py first.")
        return False
    
    # Read README.md content
    with open("README.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get the list of explanation files
    explanation_files = [f.name for f in os.scandir(explanations_dir) if f.is_file() and f.name.endswith('.md')]
    
    if not explanation_files:
        print(f"Warning: No explanation files found in {explanations_dir} directory.")
        return False
    
    # Process content line by line
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        # Skip section headers, empty lines, and dividers
        if line.startswith('#') or line.strip() == '' or line.strip() == '---':
            modified_lines.append(line)
            continue
            
        # Only process lines that look like term definitions
        if not re.match(r'^[A-Za-z]', line.strip()):
            modified_lines.append(line)
            continue
            
        # If line already has a link, don't modify it
        if line.startswith('[') and 'explanations/' in line:
            modified_lines.append(line)
            continue
            
        # Extract terms in the format "Term (한글용어)" or with ** marks
        match = re.match(r'^([^(]+)\s*(\([^)]+\))\s*(?:\*\*([^*]+)\*\*)?$', line.strip())
        if match:
            english_term = match.group(1).strip()
            korean_term = match.group(2) if match.group(2) else ""
            
            # Generate the expected filename for this term
            filename = sanitize_filename(english_term) + ".md"
            
            # Check if the explanation file exists
            if filename in explanation_files:
                # Create a link to the explanation file
                link_line = f"[{english_term}](explanations/{filename}) {korean_term}"
                
                # If there was bold text, preserve it
                if match.group(3):
                    link_line += f" **{match.group(3)}**"
                    
                modified_lines.append(link_line)
            else:
                # Keep the original line if no explanation file exists
                modified_lines.append(line)
        else:
            # Keep the original line for any non-matching lines
            modified_lines.append(line)
    
    # Determine output file
    output_file = "README.md" if update_original else "README.md.linked"
    
    # Write the modified content back to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modified_lines))
    
    # Count how many links were added
    link_count = sum(1 for line in modified_lines if line.startswith('[') and 'explanations/' in line)
    
    print(f"Added {link_count} links to explanation files")
    print(f"Updated {'README.md' if update_original else 'README.md.linked'}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add links to explanation files in README.md')
    parser.add_argument('--update-original', action='store_true', 
                        help='Update the original README.md file instead of creating README.md.linked')
    args = parser.parse_args()
    
    success = add_explanation_links(update_original=args.update_original)
    if not success:
        sys.exit(1) 