from bs4 import BeautifulSoup

def reassemble_fragments(code_to_write, file_contents):
    # Parse the original file contents
    soup = BeautifulSoup(file_contents, 'html.parser')
    
    # Parse the new fragments to insert
    new_fragments = BeautifulSoup(code_to_write, 'html.parser')
    
    # For each fragment in the new code
    for new_fragment in new_fragments.children:
        if new_fragment.name:  # Skip NavigableString objects
            # Find matching element in original soup by data-llama-id
            llama_id = new_fragment.get('data-llama-id')
            if llama_id:
                old_element = soup.find(attrs={"data-llama-id": llama_id})
                if old_element:
                    # Replace the old element with the new one
                    old_element.replace_with(new_fragment)
    
    return str(soup)