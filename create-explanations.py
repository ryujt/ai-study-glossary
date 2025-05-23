import re
import os
import requests
import time
import json
from pathlib import Path

# Configure Ollama endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest"  # Using the available gemma3:latest model

# Create directory for term explanations
OUTPUT_DIR = "explanations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_unlinked_terms_from_readme(readme_path="README.md"):
    """Extract AI terms without links from README.md file."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all terms using regex
    unlinked_terms = []
    lines = content.split('\n')
    
    for line in lines:
        # Skip section headers, empty lines, and dividers
        if line.startswith('#') or line.strip() == '' or line.strip() == '---':
            continue
        
        # Skip lines that have links (start with * [)
        if line.strip().startswith('* ['):
            continue
        
        # Process lines that are not linked
        # Format 1: Associative Memory (연상메모리)
        match = re.match(r'^([^(]+)\s*(\([^)]+\)).*$', line.strip())
        if match:
            english_term = match.group(1).strip()
            korean_term = match.group(2).strip()
            if english_term and korean_term:
                unlinked_terms.append((english_term, korean_term))
                continue
        
        # Format 2: Term (Korean)
        # This will catch other formats where the term and Korean translation are on the same line
        words = line.strip().split()
        if words and not line.strip().startswith('*') and '(' in line and ')' in line:
            # Find the term and Korean translation
            term_part = line.strip().split('(')[0].strip()
            korean_part = '(' + line.strip().split('(', 1)[1].split(')', 1)[0] + ')'
            if term_part and korean_part:
                unlinked_terms.append((term_part, korean_part))
    
    return unlinked_terms

def generate_explanation_with_ollama(term, korean_term):
    """Call Ollama API to generate an explanation for a given term."""
    prompt = f"""
    '{term} {korean_term}'에 대해 다음 형식으로 설명해주세요:

    1. 정의: 간결하게 용어를 정의해주세요.
    2. 핵심 개념: 이 용어와 관련된 핵심 개념 3-5개를 나열해주세요.
    3. 작동 방식: 어떻게 작동하는지 설명해주세요.
    4. 응용 분야: 이 개념/기술이 어디에 적용되는지 설명해주세요.
    5. 관련 용어: 연관된 다른 AI 용어 3-5개를 나열해주세요.
    
    전문가처럼 정확하지만 초보자도 이해할 수 있게 작성해주세요.
    답변 외의 문장은 절대 사용하지 말아주세요.
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "설명을 생성할 수 없습니다.")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API for term '{term}': {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return f"Ollama API 호출 중 오류가 발생했습니다: {e}"

def sanitize_filename(term):
    """Convert term to a valid filename."""
    # Remove characters that aren't allowed in filenames
    sanitized = re.sub(r'[\\/*?:"<>|]', '', term)
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[\s\-–]', '_', sanitized)
    # Remove any other problematic characters
    sanitized = re.sub(r'[^\w\._]', '', sanitized)
    return sanitized

def update_readme_with_links(terms_with_files):
    """Update README.md to add links to the generated explanation files."""
    with open("README.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Process each line to see if it contains an unlinked term
        updated_line = line
        for term, korean_term, filename in terms_with_files:
            if term in line and korean_term in line and not line.strip().startswith('* ['):
                # Create the linked format
                updated_line = f"* [{term}](explanations/{filename}) {korean_term}"
                break
        
        new_lines.append(updated_line)
    
    # Write back the updated content
    with open("README.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"README.md updated with {len(terms_with_files)} new links")

def main():
    """Main function to process README and generate explanations."""
    print("Extracting unlinked terms from README.md...")
    unlinked_terms = extract_unlinked_terms_from_readme()
    print(f"Found {len(unlinked_terms)} unlinked terms")
    
    # Filter out any remaining terms that don't match criteria
    filtered_terms = []
    for term, korean_term in unlinked_terms:
        if term and korean_term and term[0].isalpha():
            filtered_terms.append((term, korean_term))
    
    print(f"Filtered to {len(filtered_terms)} actual terms")
    
    # Skip if no terms found
    if not filtered_terms:
        print("No unlinked terms found to process.")
        return
    
    # Print the terms that will be processed
    print("Terms to be processed:")
    for term, korean_term in filtered_terms:
        print(f"- {term} {korean_term}")
    
    # First, check if Ollama is running
    try:
        test_response = requests.post(OLLAMA_API_URL, json={"model": MODEL_NAME, "prompt": "테스트", "stream": False})
        test_response.raise_for_status()
        print(f"Successfully connected to Ollama with model {MODEL_NAME}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is running and the model is available")
        return
    
    # Keep track of terms and their corresponding files
    terms_with_files = []
    
    for i, (term, korean_term) in enumerate(filtered_terms):
        clean_term = term.strip()
        filename = sanitize_filename(clean_term) + ".md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if file already exists and is not an error file
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if "Ollama API 호출 중 오류가 발생했습니다" not in content and len(content) > 200:
                print(f"[{i+1}/{len(filtered_terms)}] Skipping {clean_term} - explanation already exists")
                terms_with_files.append((clean_term, korean_term, filename))
                continue
        
        print(f"[{i+1}/{len(filtered_terms)}] Generating explanation for {clean_term} {korean_term}...")
        
        # Call Ollama to generate explanation
        explanation = generate_explanation_with_ollama(clean_term, korean_term)
        
        # Create markdown file with explanation
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {clean_term} {korean_term}\n\n")
            f.write(explanation)
        
        print(f"Saved explanation to {filepath}")
        
        # Add to list of terms with files for updating README
        terms_with_files.append((clean_term, korean_term, filename))
        
        # Be nice to the API with a short delay between requests
        time.sleep(1)
    
    # Update README.md with links to generated explanations
    if terms_with_files:
        update_readme_with_links(terms_with_files)
    
    print("All explanations generated and README updated!")

if __name__ == "__main__":
    main()
