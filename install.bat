@echo off
echo Initializing filter data submodule...
git submodule update --init --recursive

echo Setting up Python environment...
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py

