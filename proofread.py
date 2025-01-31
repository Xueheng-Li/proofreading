import os
import re
import json
from typing import List, Tuple, Dict
from langchain_openai import ChatOpenAI
from datetime import datetime
from pathlib import Path

from setting import *

PROGRESS_FILE = "proofreading_progress.json"

# generate a timestamp
def gen_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def pre_edit_text(content: str) -> str:
    """
    Replace single-line '\[' with '\begin{align*}' and
    single-line '\]' with '\end{align*}' before processing.
    """
    # substitute '.s+}' with '.}'
    content = re.sub(r'\.\s+}', '.}', content)
    # substitute '\emph{\s*}' with ' '
    content = re.sub(r'\\emph{\s*}', ' ', content)

    # add a new line after end proof, end proposition, end theorem, end lemma, end corollary, end definition, end example
    for end_mark in ['\end{proof}', '\end{prop}', '\end{proposition}', '\end{theorem}', '\end{lemma}', '\end{corollary}', '\end{definition}', '\end{example}', '\end{remark}', '\end{defn}', '\end{lem}']:
        content = content.replace(end_mark, end_mark + '\n')

    lines = content.split('\n')
    new_lines = []
    for line in lines:
        trimmed = line.strip()
        if trimmed == r'\[':
            new_lines.append(r'\begin{align*}')
        elif trimmed == r'\]':
            new_lines.append(r'\end{align*}')
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)



def analyze_latex_blocks(content: str, max_length: int = 1000) -> List[str]:
    """
    Analyze LaTeX file to find environment blocks where all instances are short.
    Handles both regular and starred (*) LaTeX environments.
    
    Args:
        file_path: Path to LaTeX file
        max_length: Maximum character length to consider a block "short"
    
    Returns:
        List of environment names where all instances are below max_length
    """
    block_stats = {}
    
    # Updated pattern to handle starred variants
    begin_pattern = r'\\begin{(.*?\*?)}' 
    begins = re.finditer(begin_pattern, content)
    
    for match in begins:
        env_name = match.group(1)
        start_pos = match.end()
        
        # Escape asterisk if present for end pattern
        escaped_env_name = env_name.replace('*', r'\*')
        end_pattern = f'\\\\end{{{escaped_env_name}}}'
        end_match = re.search(end_pattern, content[start_pos:])
        
        if end_match:
            block_length = end_match.start()
            
            if env_name not in block_stats:
                block_stats[env_name] = {'max_length': 0, 'count': 0}
            
            block_stats[env_name]['max_length'] = max(
                block_stats[env_name]['max_length'], 
                block_length
            )
            block_stats[env_name]['count'] += 1
    
    short_blocks = []
    for env_name, stats in block_stats.items():
        if stats['max_length'] < max_length:
            short_blocks.append(env_name)
    
    # Sort by max length
    short_blocks.sort(key=lambda x: block_stats[x]['max_length'])
    
    print("\nBlock statistics:")
    for env_name in short_blocks:
        stats = block_stats[env_name]
        print(f"{env_name}: max_length={stats['max_length']}, count={stats['count']}")
    
    return short_blocks



class LaTeXProofreader:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            base_url=BASE_URL,
            api_key=API_KEY
        )
        
        self.system_prompt = SYSTEM_PROMPT
        
        # Update skip commands to be more specific
        self.skip_commands = {
            r'^\s*\\section{.*?}$',
            r'^\s*\\subsection{.*?}$',
            r'^\s*\\begin{.*?}$',
            r'^\s*\\end{.*?}$',
            r'^\s*\\item\s*.*$',
            r'^\s*\\bibliography{.*?}$',
            r'^\s*\\bibliographystyle{.*?}$'
        }
        
        # Commands that are allowed within text - update citation patterns
        self.inline_commands = {
            r'\\emph{.*?}',
            r'\\cite(?:[tp])?\s*(?:\[.*?\])?\s*(?:\[.*?\])?\s*{.*?}',  # handles \cite, \citep, \citet with optional args
            r'\\citeyear(?:\s*\[.*?\])?\s*{.*?}',  # handles \citeyear with optional args
            r'{\\.*?{.*?}}'
        }

        # Add citation command prefixes for detection
        self.citation_prefixes = {'\\cite', '\\citep', '\\citet', '\\citeyear'}
        self.current_file = None
        self.file_lines = []
        self.section_text = ""

        # Add skip blocks
        self.skip_blocks = {
            'figure',
            'table',
            'equation',
            'equation*',
            'align',
            'align*',
            'tabular'
        }

    def is_citation_command(self, text: str) -> bool:
        """Check if text starts with a citation command."""
        return any(text.strip().startswith(prefix) for prefix in self.citation_prefixes)

    def is_primarily_command(self, text: str) -> bool:
        """Check if the text is primarily a LaTeX command rather than content."""
        text = text.strip()
        # Always process citations, even standalone ones
        if self.is_citation_command(text):
            return False
            
        # Skip if text is just a command
        if text.startswith('\\') and not any(cmd in text for cmd in ['\\cite', '\\emph']):
            words = text.split()
            # If first word is a command and there's little other content
            if words and words[0].startswith('\\') and len(words) <= 3:
                return True
        return False

    def should_process_text(self, text: str) -> bool:

        # process if multiple lines; check if text contains multiple lines
        if len(text.split('\n')) > 1:
            return True

        """Check if text should be processed or returned as-is."""
        if not text.strip():
            return False
            
        # Skip if text is just a command
        if self.is_primarily_command(text):
            return False
            
        # Skip if text exactly matches any of the skip patterns
        for pattern in self.skip_commands:
            if re.match(pattern, text.strip()):
                return False

        # Check if text is within a skip block
        for block in self.skip_blocks:
            if text.strip().startswith(f"\\begin{{{block}}}"):
                print(f"Skipping block: {block}")
                print(text)
                return False

        # Skip if text is too short; likely a command or special text
        if len(text.strip()) < 15:
            return False
        
        return True

    def extract_main_text(self, content: str) -> str:
        """Extract text between \begin{document} and \end{document}."""
        start = content.find("\\begin{document}")
        end = content.find("\\end{document}")
        if (start == -1 or end == -1):
            raise ValueError("Could not find \\begin{document} or \\end{document} tags in the file")
        # Include text after \begin{document} but before \end{document}
        return content[start + len("\\begin{document}"):end].strip()

    def find_line_number(self, text_start: str, start_from: int = 0) -> int:
        """Find the line number where the text starts in the current file."""
        if not text_start.strip():
            return start_from
            
        # Look for the first non-empty line in the text
        first_line = next((line for line in text_start.split('\n') if line.strip()), '')
        if not first_line:
            return start_from

        # Search through file lines starting from the given position
        for i in range(start_from, len(self.file_lines)):
            if first_line.strip() in self.file_lines[i]:
                return i + 1  # Line numbers are 1-based
        return start_from
    


    def split_into_paragraphs(self, content: str, short_blocks: list = []) -> List[Tuple[str, str, int, bool]]:
        """Split LaTeX content into paragraphs based on blank lines and LaTeX environments."""
        # Extract main text
        main_text = self.extract_main_text(content)
        
        # Store file lines for line number lookup
        self.file_lines = content.split('\n')
        
        # Split into paragraphs using blank lines
        paragraphs = []
        current_para = []
        para_count = 0
        command_count = 0
        lines = main_text.split('\n')
        last_line_number = self.find_line_number("\\begin{document}", 1)  # Start from line 1
        i = 0

        block_protect_list0 = ['lem', 'lemma', 'rem', 'theorem', 'prop', 'definition','remark', 'abstract', 'proposition', 'figure', 'table', 'quotation', 'thm', 'defn', 'assumption', 'claim', 'align', 'equation']
        block_protect_list = []
        # add block\* to extend the list
        for block_name in block_protect_list0:
            block_protect_list.append(block_name)
            block_protect_list.append(block_name + '\*')
        print(f"Block protect list: {block_protect_list}")
        for block_name in short_blocks:
            block_name = block_name.replace('*', '\*').replace('\\*', '\*')
            if block_name not in block_protect_list:
                block_protect_list.append(block_name)
        print(f"Block protect list: {block_protect_list}")
        
        while i < len(lines):
            line = lines[i].strip()
            original_line = lines[i]  # Keep the original line with whitespace

            # For equation, quotation, proposition... blocks, process them as a whole unit
            block_found = False
            for block_name in block_protect_list:
                block_start_pattern = rf'\\begin{{{block_name}}}' if block_name not in ['\['] else r'\\\['
                block_end_pattern = rf'\\end{{{block_name}}}' if block_name not in ['\['] else r'\\\]'
                
                if re.match(block_start_pattern, line):
                    block_found = True
                    # Handle current paragraph if exists
                    if current_para:
                        para_text = '\n'.join(current_para)
                        line_number = self.find_line_number(para_text, last_line_number)
                        if self.should_process_text(para_text):
                            para_count += 1
                            paragraphs.append((f"Paragraph {para_count}", para_text, line_number, False))
                        else:
                            command_count += 1
                            paragraphs.append((f"Command {command_count}", para_text, line_number, False))
                        last_line_number = line_number
                        current_para = []
                    
                    # Collect block content until matching end
                    block_lines = [original_line]
                    i += 1
                    block_complete = False
                    while i < len(lines):
                        block_lines.append(lines[i])
                        if re.match(block_end_pattern, lines[i].strip()):
                            block_complete = True
                            break
                        i += 1
                    
                    if block_complete:
                        block_text = '\n'.join(block_lines)
                        line_number = self.find_line_number(block_text, last_line_number)
                        command_count += 1
                        block_name = "Math" if block_name == '\[' else block_name.capitalize()
                        paragraphs.append((f"{block_name} {command_count}", block_text, line_number, False))
                        last_line_number = line_number
                        i += 1
                    break
                
            if block_found:
                continue

            # Check for LaTeX environment start
            if re.match(r'\\begin{.*?}', line):
                if current_para:
                    para_text = '\n'.join(current_para)
                    line_number = self.find_line_number(para_text, last_line_number)
                    if self.should_process_text(para_text):
                        para_count += 1
                        paragraphs.append((f"Paragraph {para_count}", para_text, line_number, False))
                    else:
                        command_count += 1
                        paragraphs.append((f"Command {command_count}", para_text, line_number, False))
                    last_line_number = line_number
                    current_para = []
                line_number = self.find_line_number(line, last_line_number)
                paragraphs.append((f"Command {command_count + 1}", line, line_number, False))
                command_count += 1
                last_line_number = line_number
                i += 1
                continue
            
            # Check for LaTeX environment end
            if re.match(r'\\end{.*?}', line):
                if current_para:
                    para_text = '\n'.join(current_para)
                    line_number = self.find_line_number(para_text, last_line_number)
                    if self.should_process_text(para_text):
                        para_count += 1
                        paragraphs.append((f"Paragraph {para_count}", para_text, line_number, False))
                    else:
                        command_count += 1
                        paragraphs.append((f"Command {command_count}", para_text, line_number, False))
                    last_line_number = line_number
                    current_para = []
                line_number = self.find_line_number(line, last_line_number)
                paragraphs.append((f"Command {command_count + 1}", line, line_number, False))
                command_count += 1
                last_line_number = line_number
                i += 1
                continue
            
            
            # Skip blank lines between paragraphs
            if not line:
                if current_para:
                    para_text = '\n'.join(current_para)
                    line_number = self.find_line_number(para_text, last_line_number)
                    if self.should_process_text(para_text):
                        para_count += 1
                        paragraphs.append((f"Paragraph {para_count}", para_text, line_number, False))
                    else:
                        command_count += 1
                        paragraphs.append((f"Command {command_count}", para_text, line_number, False))
                    last_line_number = line_number
                    current_para = []
                # Add empty line as a special unit
                paragraphs.append((f"Empty {i}", original_line, i + 1, True))
                i += 1
                continue
            
            # Add line to current paragraph
            current_para.append(original_line)  # Keep original spacing
            i += 1
        
        # Add the last paragraph if not empty
        if current_para:
            para_text = '\n'.join(current_para)
            line_number = self.find_line_number(para_text, last_line_number)
            if self.should_process_text(para_text):
                para_count += 1
                paragraphs.append((f"Paragraph {para_count}", para_text, line_number, False))
            else:
                command_count += 1
                paragraphs.append((f"Command {command_count}", para_text, line_number, False))
            
        return paragraphs


    def post_edit(self, text: str) -> str:

        # sometimes LLM returns results enclosed in ```latex ... ``` or ```...```; extract the content using re
        match = re.match(r'```(?:latex)?\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1)

        # Post-processing step to handle $ signs - check if original has any math mode 
        # a function check if self.section_text has any math mode symbols at all
        def has_math_mode(text: str) -> bool:
            return any(sym in text for sym in ['$', '\\(', '\\)'])
        
        if not has_math_mode(self.section_text) and has_math_mode(text):
            """If the original text has no math mode, but the edited text does, delete the math mode."""
            text = re.sub(r'(\$|\\(|\\))', '', text)

        ends_with_endproof = self.section_text.strip().endswith('\end{proof}')
        if not ends_with_endproof and text.strip().endswith('\end{proof}'):
            text = text.rstrip('\end{proof}').rstrip()

        if len(text.split('\n')) <= 3:
            """Post-edit text to put each sentence on a new line."""
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Za-z])', text)
            return '\n'.join(sentences)
        
        # strip any ending one or multiple of /n or space from the text
        text = text.rstrip('\n ')
        
        return text


    def proofread_section(self, section_text: str, stream: bool = False) -> str:
        """Proofread a single section of the document."""
        # Return text as-is if it shouldn't be processed
        if not self.should_process_text(section_text):
            print("Skipping LaTeX command or special text:")
            print(section_text)
            return section_text
        # # replace \n with space
        # content = content.replace('\n', ' ')
        # Store the section text for post-editing
        self.section_text = section_text
        
        # Prepare messages for the chat model
        messages = f"{self.system_prompt}\n\nHere is the text for you to copy-edit:\n\n{section_text}\n\nReturn the complete copy-edited text without comments or explanations."
        
        print(f"Input text:\n{'-'*50}\n{section_text}\n{'-'*50}")
        
        try:
            print(f"Output text:\n{'-'*50}")
            # Get streaming response from the model
            if stream:
                full_response = ""
                for chunk in self.llm.stream(messages):
                    if chunk and hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                print(f"\n{'-'*50}")
            else:
                full_response = self.llm.invoke(messages)
                full_response = full_response.content
            
            if not full_response:
                print("\nNo response received from model")
                return section_text  # Return original text if no response
            
            # Replace \( ... \) with $ ... $ to keep LaTeX grammar
            full_response = re.sub(r'\\\((.*?)\\\)', r'$\1$', full_response)
            
            # Post-edit
            full_response = self.post_edit(full_response)
            
            if not stream:
                print(f"{full_response}\n{'-'*50}")

            # if full_response differs from section_text too much by length ratio, print both and skip it
            if len(full_response) > 1.5 * len(section_text) or len(full_response) < 0.5 * len(section_text):
                print("Skipping this text due to large changes")
                return section_text
                
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return section_text  # Return original text on error
        
        return full_response

    def load_progress(self, file_path: str) -> Dict:
        """Load progress from progress file."""
        progress_path = Path(PROGRESS_FILE)
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                if progress.get('file_path') == file_path:
                    return progress
        return {'file_path': file_path, 'completed_units': [], 'timestamp': None}

    def save_progress(self, progress: Dict) -> None:
        """Save progress to progress file."""
        progress['timestamp'] = datetime.now().isoformat()
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)

    def append_to_output(self, output_path: str, content: str, first_write: bool = False) -> None:
        """Append content to output file."""
        mode = 'w' if first_write else 'a'
        with open(output_path, mode, encoding='utf-8') as f:
            if not first_write:
                # Add single newline between sections
                f.write('\n')
            f.write(content)

    def proofread_document(self, file_path: str, output_path: str, resume: bool = True, stream: bool = False) -> None:
        """Proofread entire LaTeX document with progress tracking and resume capability."""
        self.current_file = file_path
        try:
            # Delete output file if not resuming and file exists
            if not resume and os.path.exists(output_path):
                os.remove(output_path)
                print(f"Deleted existing output file: {output_path}")
                
            # Load progress if resuming
            progress = self.load_progress(file_path) if resume else {
                'file_path': file_path,
                'completed_units': [],
                'timestamp': None
            }
            
            # Read the input file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # analyze LaTeX blocks to find short blocks
            short_blocks = analyze_latex_blocks(original_content, max_length=1000)
            
            # Pre-edit the entire content
            content = pre_edit_text(original_content)
            
            # Split into paragraphs
            units = self.split_into_paragraphs(content, short_blocks=short_blocks)
            
            # Initialize output file if not resuming
            if not resume or not os.path.exists(output_path):
                # Write the document beginning
                doc_start = content[:content.find("\\begin{document}") + len("\\begin{document}")]
                self.append_to_output(output_path, doc_start, True)
            
            # Process each unit
            for title, unit_text, line_number, is_empty in units:
                # Skip if already processed
                if title in progress['completed_units']:
                    print(f"Skipping completed unit: {title}")
                    continue
                    
                print("\n" + "="*40 + f"  {gen_timestamp()}  " + "="*20)
                print(f"Processing {title} (line {line_number})")
                
                # If it's an empty line, append it as-is
                if is_empty:
                    self.append_to_output(output_path, unit_text)
                else:
                    # Proofread the unit
                    proofread_text = self.proofread_section(unit_text, stream=stream)
                    self.append_to_output(output_path, proofread_text)
                
                # Update progress
                progress['completed_units'].append(title)
                self.save_progress(progress)
                print(f"Progress saved. Completed {len(progress['completed_units'])}/{len(units)} units")
            
            # Write the document ending
            doc_end = content[content.find("\\end{document}"):]
            self.append_to_output(output_path, doc_end)
            
            print(f"Proofreading complete. Output saved to {output_path}")
            
            # Clear progress file if completed
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
            
        except Exception as e:
            print(f"Error during proofreading: {str(e)}")
            raise

class Tee:
    def __init__(self, *fileobjs):
        self.fileobjs = fileobjs

    def write(self, text):
        for f in self.fileobjs:
            f.write(text)

    def flush(self):
        for f in self.fileobjs:
            f.flush()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Proofread LaTeX document')
    parser.add_argument('--input', type=str, required=True, help='Path to the input LaTeX file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output LaTeX file')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignoring any saved progress')
    parser.add_argument('--stream', action='store_true', help='Stream model output in real-time')
    args = parser.parse_args()
    
    # Initialize proofreader
    proofreader = LaTeXProofreader(model_name=MODEL_NAME)
    
    # Define input and output paths
    input_file = args.input
    output_file = args.output
    
    import sys
    log_file = output_file.replace('.tex', '_proofread.log')
    # if no resume and log file exists, delete it
    if args.no_resume and os.path.exists(log_file):
        os.remove(log_file)
    # append to log file
    with open(log_file, 'a', encoding='utf-8') as log:
        orig_stdout = sys.stdout
        sys.stdout = Tee(orig_stdout, log)
        try:
            # Process the document
            proofreader.proofread_document(
                input_file,
                output_file,
                resume=not args.no_resume,
                stream=args.stream
            )
        finally:
            sys.stdout = orig_stdout

if __name__ == "__main__":
    main()
