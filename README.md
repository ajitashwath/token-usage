# Token Usage Visualizer
A web application that provides instant tokenization analysis and cost estimation across multiple Large Language Models (LLMs). Perfect for developers, researchers, and businesses optimizing their LLM usage and costs.

## ✨ Key Features
- **🔍 Multi-Model Analysis** - Compare tokenization across GPT-3.5, GPT-4, Claude, Gemini, and Mistral
- **💰 Real-Time Cost Estimation** - Instant pricing for input/output tokens across 9 LLM providers
- **📊 Visual Analytics** - Token breakdown tables, length distribution charts, and comprehensive statistics
- **🚀 Blazing Fast** - Completely offline with optimized caching and performance
- **📁 Flexible Input** - Direct text input or file upload (supports .txt, .md, .py, .json, .csv)

## 🎯 Perfect For
- **Developers** optimizing prompt engineering and token usage
- **Product Teams** estimating LLM integration costs
- **Researchers** analyzing tokenization patterns across models
- **Businesses** making informed decisions about LLM provider selection

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Open browser to http://localhost:8501
```

## 🛠️ Core Dependencies
- `streamlit` - Web interface
- `tiktoken` - OpenAI tokenization
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations

## 🔧 Usage
1. **Select Tokenizer** - Choose encoding (cl100k_base for GPT-4/3.5-turbo)
2. **Input Text** - Type directly or upload files
3. **Analyze** - View tokens, costs, and statistics instantly

## 📈 What You Get
- **Token Metrics** - Count, character ratio, processing time
- **Cost Breakdown** - Per-model pricing with cheapest/most expensive highlights
- **Detailed Analysis** - Token-by-token breakdown with IDs and types
- **Visual Insights** - Length distribution and statistical summaries
- **Performance Stats** - Unique tokens, type classification, processing metrics
