from __future__ import annotations

import re
from typing import Dict


def parse_llm_output(raw_output: str, parse_code_subsections: bool = False) -> Dict[str, str]:
    """
    Parse LLM output into sections based on the format specified in system_prompt.txt.
    Expected sections using LaTeX \\section commands: Problem, Problem Description, Answer Requirements, Solution, Answer, Code
    Also supports legacy formats: A) Problem, B) Answer Requirements, etc.
    
    Args:
        raw_output: The raw LLM output to parse
        parse_code_subsections: If True, expects Code section to have \\subsection{Task N} markers
                                and will extract only the first subsection. If False (default),
                                extracts the entire Code section content.
    
    Returns a dict with keys: "Problem Statement", "Problem Description", "Answer Requirements", "Solution", "Answer", "Code"
    Missing sections will be empty strings. Section headers are excluded from content.
    """
    result: Dict[str, str] = {
        "Problem Statement": "",
        "Problem Description": "",
        "Answer Requirements": "",
        "Solution": "",
        "Answer": "",
        "Code": "",
    }
    
    text = raw_output
    
    # Primary: Look for LaTeX \section{...} markers (preferred format)
    latex_section_pattern = r'\\section\{([^}]+)\}'
    section_mapping = {
        "Problem": "Problem Statement",
        "Problem Description": "Problem Description",
        "Answer Requirements": "Answer Requirements",
        "Solution": "Solution",
        "Answer": "Answer",
        "Code": "Code",
    }
    
    section_positions = {}  # Maps section name -> (start_pos, end_pos)
    
    # Find all LaTeX section markers
    for match in re.finditer(latex_section_pattern, text):
        section_title = match.group(1).strip()
        if section_title in section_mapping:
            mapped_name = section_mapping[section_title]
            # Start position is after the header line (skip the \section{...} line entirely)
            start_line_end = text.find('\n', match.end())
            if start_line_end == -1:
                start_pos = len(text)
            else:
                start_pos = start_line_end + 1
            section_positions[mapped_name] = (start_pos, None)  # end_pos will be set later
    
    # If LaTeX sections found, use them
    if section_positions:
        # Sort sections by position
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1][0])
        
        # Set end positions
        for i, (section_name, (start_pos, _)) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                # End before next section starts
                next_start = sorted_sections[i + 1][1][0]
            else:
                # Last section goes to end
                next_start = len(text)
            
            # Extract content, making sure to exclude any trailing section headers
            content = text[start_pos:next_start].strip()
            
            # Remove any trailing section headers that might have been included
            # Look for patterns like "### B) \textbf{Answer Requirements}" at the end
            content = re.sub(
                r'(?:#{1,4}\s*[A-E]\)\s*[\\textbf{]*[^}]+\}?\s*)+$',
                '',
                content,
                flags=re.MULTILINE
            )
            content = re.sub(r'\\section\{[^}]+\}\s*$', '', content, flags=re.MULTILINE)
            
            # Remove code block markers if present
            content = re.sub(r'^```(?:python|py|)?\n', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
            
            # Special handling for Code section: optionally parse subsections
            if section_name == "Code" and parse_code_subsections:
                # Extract only the first subsection if subsections are present
                subsection_pattern = r'\\subsection\{[^}]+\}'
                subsection_match = re.search(subsection_pattern, content)
                if subsection_match:
                    # Find the start of the first subsection
                    subsection_start = subsection_match.end()
                    # Find the next subsection or end of content
                    next_subsection = re.search(subsection_pattern, content[subsection_start:])
                    if next_subsection:
                        # Extract only the first subsection content
                        content = content[subsection_start:subsection_start + next_subsection.start()].strip()
                    else:
                        # Extract from first subsection to end
                        content = content[subsection_start:].strip()
                    # Remove any trailing LaTeX commands
                    content = re.sub(r'\\subsection\{[^}]+\}\s*$', '', content, flags=re.MULTILINE)
            
            result[section_name] = content.strip()
        
        return result
    
    # Fallback: Legacy format patterns (A) Problem, B) Answer Requirements, etc.)
    legacy_patterns = {
        "Problem Statement": [
            r'#{0,4}\s*A\)\s*[\\textbf{]*Problem[}\s]*:?\s*\n',
            r'\*\*A\)\s*Problem\*\*',
            r'A\)\s*Problem\s*\n',
            r'\\section\{A\)\s*Problem\}',
        ],
        "Answer Requirements": [
            r'#{0,4}\s*B\)\s*[\\textbf{]*Answer\s*Requirements[}\s]*:?\s*\n',
            r'\*\*B\)\s*Answer\s*Requirements\*\*',
            r'B\)\s*Answer\s*Requirements\s*\n',
            r'\\section\{B\)\s*Answer\s*Requirements\}',
        ],
        "Solution": [
            r'#{0,4}\s*C\)\s*[\\textbf{]*Solution[}\s]*:?\s*\n',
            r'\*\*C\)\s*Solution\*\*',
            r'C\)\s*Solution\s*\n',
            r'\\section\{C\)\s*Solution\}',
        ],
        "Answer": [
            r'#{0,4}\s*D\)\s*[\\textbf{]*Answer[}\s]*:?\s*\n',
            r'\*\*D\)\s*Answer\*\*',
            r'D\)\s*Answer\s*\n',
            r'\\section\{D\)\s*Answer\}',
        ],
        "Code": [
            r'#{0,4}\s*E\)\s*[\\textbf{]*Code[}\s]*:?\s*\n',
            r'\*\*E\)\s*Code\*\*',
            r'E\)\s*Code\s*\n',
            r'\\section\{E\)\s*Code\}',
        ],
    }
    
    section_starts = {}
    for section_name, pattern_list in legacy_patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Start after the header line
                start_line_end = text.find('\n', match.end())
                if start_line_end == -1:
                    section_starts[section_name] = len(text)
                else:
                    section_starts[section_name] = start_line_end + 1
                break
    
    # Extract sections in order
    section_order = ["Problem Statement", "Problem Description", "Answer Requirements", "Solution", "Answer", "Code"]
    for i, section_name in enumerate(section_order):
        if section_name in section_starts:
            start_pos = section_starts[section_name]
            # Find the start of the next section or end of text
            next_start = len(text)
            for next_section in section_order[i + 1:]:
                if next_section in section_starts:
                    next_start = section_starts[next_section]
                    break
            
            content = text[start_pos:next_start].strip()
            
            # Remove trailing section headers that might have been included
            content = re.sub(
                r'(?:#{1,4}\s*[A-E]\)\s*[\\textbf{]*[^}]+\}?\s*)+$',
                '',
                content,
                flags=re.MULTILINE
            )
            content = re.sub(r'\\section\{[^}]+\}\s*$', '', content, flags=re.MULTILINE)
            
            # Remove code block markers if present
            content = re.sub(r'^```(?:python|py|)?\n', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
            
            # Special handling for Code section: optionally parse subsections
            if section_name == "Code" and parse_code_subsections:
                # Extract only the first subsection if subsections are present
                subsection_pattern = r'\\subsection\{[^}]+\}'
                subsection_match = re.search(subsection_pattern, content)
                if subsection_match:
                    # Find the start of the first subsection
                    subsection_start = subsection_match.end()
                    # Find the next subsection or end of content
                    next_subsection = re.search(subsection_pattern, content[subsection_start:])
                    if next_subsection:
                        # Extract only the first subsection content
                        content = content[subsection_start:subsection_start + next_subsection.start()].strip()
                    else:
                        # Extract from first subsection to end
                        content = content[subsection_start:].strip()
                    # Remove any trailing LaTeX commands
                    content = re.sub(r'\\subsection\{[^}]+\}\s*$', '', content, flags=re.MULTILINE)
            
            result[section_name] = content.strip()
    
    # Final fallback: if no structured parsing worked, put everything in Problem Statement
    if not any(result.values()):
        result["Problem Statement"] = text.strip()
    
    return result

