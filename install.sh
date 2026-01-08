#!/bin/bash
echo "Initializing filter data submodule..."
git submodule update --init --recursive

echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r program/requirements.txt

streamlit run program/app.py
