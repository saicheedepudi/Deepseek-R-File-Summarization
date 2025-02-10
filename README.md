# PDF Summarizer App

A Streamlit-based application that automatically generates concise summaries of PDF documents using Ollama and DeepSeek.

## Overview

Key technical components:
- **DeepSeek-R1 1.5B** - Reasoning-optimized LLM for document understanding and summarization
- **Ollama** - Local LLM execution framework for private, offline processing
- **Streamlit** - Web interface for PDF upload and summary display

## Features
- 📄 PDF file upload interface
- 🧠 Text chunking and processing
- ✨ AI-powered document summarization
- ⚡ Real-time processing feedback
- 🔍 Optional chunk inspection


![Summarize File](https://github.com/user-attachments/assets/bfad36ce-93cb-4053-9900-6247a2a939b4)


<img width="884" alt="image" src="https://github.com/user-attachments/assets/ae7c9623-9355-4f47-9f34-25e22c99579d" />

## Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Deepseek model: `ollama pull deepseek-r1:1.5b`

## Installation
Pull DeepSeek model
 ollama pull deepseek-r1:1.5b

## Setup

Install dependencies:

pip install -r requirements.txt

Create PDF directory:

mkdir -p pdf

streamlit run main.py
