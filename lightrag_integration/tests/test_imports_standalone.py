#!/usr/bin/env python3
"""
Standalone import test to validate the query classification system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def test_imports():
    """Test all imports work correctly."""
    try:
        # Import cost_persistence first
        import cost_persistence
        print(f"✓ cost_persistence imported: {cost_persistence}")
        
        # Import ResearchCategory
        from cost_persistence import ResearchCategory
        print(f"✓ ResearchCategory imported: {list(ResearchCategory)[:3]}...")
        
        # Temporarily fix the relative import issue
        with open('research_categorizer_temp.py', 'w') as f:
            with open('../research_categorizer.py', 'r') as original:
                content = original.read()
                # Replace the relative import with absolute import
                fixed_content = content.replace(
                    'from .cost_persistence import ResearchCategory',
                    'from cost_persistence import ResearchCategory'
                )
                f.write(fixed_content)
        
        # Import the fixed version
        import research_categorizer_temp as research_categorizer
        print(f"✓ research_categorizer imported: {research_categorizer}")
        
        # Import specific classes
        from research_categorizer_temp import (
            ResearchCategorizer,
            CategoryPrediction,
            QueryAnalyzer,
            CategoryMetrics
        )
        print("✓ All research_categorizer classes imported")
        
        # Test instantiation
        categorizer = ResearchCategorizer()
        print("✓ ResearchCategorizer instantiated")
        
        # Test categorization
        result = categorizer.categorize_query("LC-MS metabolite identification")
        print(f"✓ Categorization works: {result.category}, confidence: {result.confidence:.3f}")
        
        # Cleanup
        import os
        os.remove('research_categorizer_temp.py')
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            import os
            if os.path.exists('research_categorizer_temp.py'):
                os.remove('research_categorizer_temp.py')
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_imports()
    print(f"\nImport validation: {'✓ SUCCESS' if success else '✗ FAILED'}")
    sys.exit(0 if success else 1)