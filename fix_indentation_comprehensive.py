#!/usr/bin/env python3
"""
Comprehensive script to fix indentation issues in the async test methods
"""

def fix_async_test_indentation():
    file_path = "lightrag_integration/tests/test_pdf_processor.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # First, remove trailing spaces and inconsistent indentation
    lines = content.split('\n')
    fixed_lines = []
    
    in_async_method = False
    method_indent = 0
    
    for i, line in enumerate(lines):
        # Remove trailing spaces
        line = line.rstrip()
        
        # Check if we're starting an async method
        if "async def run_test():" in line:
            in_async_method = True
            method_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        # Check if we're ending the async method
        if in_async_method and line.strip() == "asyncio.run(run_test())":
            in_async_method = False
            fixed_lines.append(line)
            continue
        
        if in_async_method:
            if line.strip() == "":
                fixed_lines.append("")
                continue
            
            # Calculate base indentation for async method content
            base_indent = method_indent + 12  # 3 levels from method start
            
            # Handle different types of lines
            if line.strip().startswith("with tempfile.TemporaryDirectory"):
                fixed_lines.append(" " * base_indent + line.strip())
            elif line.strip().endswith("= Path(tmp_dir)"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("#"):
                # Comments should be aligned with the code block they describe
                if i+1 < len(lines) and lines[i+1].strip().startswith("with patch"):
                    fixed_lines.append(" " * (base_indent + 4) + line.strip())
                else:
                    fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("with patch"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("mock_"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("pdf_files"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("large_content"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("self.processor"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("def mock_"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("successful_docs"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("for i in range"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("result ="):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("assert"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("texts ="):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("log_"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("error_"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("info_"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("start_time"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("end_time"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("for"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("required_fields"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("text, metadata"):
                fixed_lines.append(" " * (base_indent + 16) + line.strip())
            elif line.strip().startswith("item"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif "in result:" in line.strip():
                fixed_lines.append(" " * (base_indent + 16) + line.strip())
            elif "in range(" in line.strip():
                fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif line.strip().startswith("return"):
                if "test_paper_" in line:
                    fixed_lines.append(" " * (base_indent + 12) + line.strip())
                else:
                    fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif line.strip().startswith("if ") or line.strip().startswith("elif ") or line.strip().startswith("else"):
                if "test_paper_" in line:
                    fixed_lines.append(" " * (base_indent + 8) + line.strip())
                elif "field in required_fields" in line:
                    fixed_lines.append(" " * (base_indent + 16) + line.strip())
                else:
                    fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif line.strip().startswith("raise"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("path"):
                fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif line.strip().startswith("doc"):
                fixed_lines.append(" " * (base_indent + 12) + line.strip())
            elif line.strip().startswith("mock_doc"):
                fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif line.strip().startswith("mock_page"):
                fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif ".append(" in line:
                fixed_lines.append(" " * (base_indent + 8) + line.strip())
            elif line.strip().startswith("non_existent_dir"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            elif line.strip().startswith("(papers_dir"):
                fixed_lines.append(" " * (base_indent + 4) + line.strip())
            else:
                # Default case - try to maintain reasonable indentation
                if line.strip():
                    fixed_lines.append(" " * (base_indent + 4) + line.strip())
                else:
                    fixed_lines.append("")
        else:
            fixed_lines.append(line)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write('\n'.join(fixed_lines) + '\n')
    
    print(f"Comprehensively fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_async_test_indentation()