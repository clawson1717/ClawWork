import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_structure():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    assert os.path.exists(os.path.join(base_dir, 'src'))
    assert os.path.exists(os.path.join(base_dir, 'tests'))
    assert os.path.exists(os.path.join(base_dir, 'data'))
    assert os.path.exists(os.path.join(base_dir, 'requirements.txt'))

def test_imports():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import src
    assert src.__version__ == "0.1.0"
