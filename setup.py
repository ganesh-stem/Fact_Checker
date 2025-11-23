"""
Setup script for AI Fact Checker

Handles installation and initial setup.
"""

import os
import subprocess
import sys


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_spacy_model():
    """Download spaCy English model."""
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def create_env_file():
    """Create .env file from example if it doesn't exist."""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("Creating .env file from template...")
        with open(".env.example", "r") as src:
            with open(".env", "w") as dst:
                dst.write(src.read())
        print("Created .env file. Please edit it to add your API keys if needed.")


def verify_installation():
    """Verify that all components are properly installed."""
    print("\nVerifying installation...")

    errors = []

    # Check spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("[PASS] spaCy installed and model loaded")
    except Exception as e:
        errors.append(f"spaCy: {e}")
        print(f"[FAIL] spaCy: {e}")

    # Check sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[PASS] sentence-transformers installed")
    except Exception as e:
        errors.append(f"sentence-transformers: {e}")
        print(f"[FAIL] sentence-transformers: {e}")

    # Check FAISS
    try:
        import faiss
        print("[PASS] faiss-cpu installed")
    except Exception as e:
        errors.append(f"faiss: {e}")
        print(f"[FAIL] faiss: {e}")

    # Check Streamlit
    try:
        import streamlit
        print("[PASS] streamlit installed")
    except Exception as e:
        errors.append(f"streamlit: {e}")
        print(f"[FAIL] streamlit: {e}")

    # Check data file
    if os.path.exists("data/verified_facts.csv"):
        print("[PASS] Data file found")
    else:
        errors.append("data/verified_facts.csv not found")
        print("[FAIL] Data file not found")

    if errors:
        print(f"\n[WARN] Setup completed with {len(errors)} warning(s)")
    else:
        print("\n[PASS] All components verified successfully!")

    return len(errors) == 0


def main():
    """Main setup function."""
    print("=" * 50)
    print("AI Fact Checker - Setup")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\nStep 1: Installing requirements...")
    try:
        install_requirements()
    except subprocess.CalledProcessError as e:
        print(f"Warning: Some packages may not have installed: {e}")

    print("\nStep 2: Downloading spaCy model...")
    try:
        download_spacy_model()
    except subprocess.CalledProcessError as e:
        print(f"Warning: spaCy model download failed: {e}")

    print("\nStep 3: Creating environment file...")
    create_env_file()

    print("\nStep 4: Verifying installation...")
    success = verify_installation()

    print("\n" + "=" * 50)
    if success:
        print("Setup completed successfully!")
        print("\nTo start the application:")
        print("  streamlit run app.py")
        print("  or")
        print("  python main.py --interactive")
    else:
        print("Setup completed with warnings.")
        print("Please resolve the issues above before running.")
    print("=" * 50)


if __name__ == "__main__":
    main()
