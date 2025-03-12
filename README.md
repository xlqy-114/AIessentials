# AIessentials
AI essential class final project source code

# Create Visual Environment
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install streamlit pymupdf pdfplumber langchain-text-splitters openai numpy python-dotenv scikit-learn
pip install --force-reinstall langchain-text-splitters

# Environment Check
python -c "import streamlit, fitz, pdfplumber, sklearn; print('Success')"

# Cuda
pip install cupy-cuda11x
export SKLEARN_ENABLE_GPU=1



