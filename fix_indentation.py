#!/usr/bin/env python3
"""
Script to fix indentation issues in the test file
"""

def fix_indentation():
    file_path = "lightrag_integration/tests/test_pdf_processor.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_async_method = False
    for i, line in enumerate(lines):
        # Detect start of async methods in batch processing class
        if "async def run_test():" in line:
            in_async_method = True
            fixed_lines.append(line)
            continue
        
        # End of async method
        if in_async_method and line.strip() == "asyncio.run(run_test())":
            in_async_method = False
            fixed_lines.append(line)
            continue
        
        # Fix indentation issues in async methods
        if in_async_method:
            # Lines that should be indented to 12 spaces (3 levels from method start)
            if line.startswith("            papers_dir = Path(tmp_dir)"):
                fixed_lines.append("                papers_dir = Path(tmp_dir)\n")
            elif line.startswith("            # ") and "with tempfile.TemporaryDirectory" not in lines[i-1]:
                fixed_lines.append("                " + line[12:])
            elif line.startswith("            ") and not line.strip().startswith("with tempfile.TemporaryDirectory"):
                # Most content should be indented to at least 16 spaces (4 levels)
                fixed_lines.append("                " + line[12:])
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation()