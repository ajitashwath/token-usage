import streamlit as st
import tiktoken
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
from typing import List, Dict, Tuple, Optional

st.set_page_config(page_title = "Token Usage", layout = "wide", initial_sidebar_state = "expanded")

# Own research
LLM_PRICING = {
    "GPT-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "GPT-4": {"input": 0.03, "output": 0.06},
    "GPT-4-turbo": {"input": 0.01, "output": 0.03},
    "Claude-3-Haiku": {"input": 0.00025, "output": 0.00125},
    "Claude-3-Sonnet": {"input": 0.003, "output": 0.015},
    "Claude-3-Opus": {"input": 0.015, "output": 0.075},
    "Gemini-Pro": {"input": 0.0005, "output": 0.0015},
    "Mistral-7B": {"input": 0.0002, "output": 0.0002},
    "Mistral-Medium": {"input": 0.0027, "output": 0.0081}
}

TIKTOKEN_ENCODINGS = [
    "cl100k_base",  # GPT-4, GPT-3.5-turbo
    "p50k_base",    # Codex
    "r50k_base",    # GPT-3 
    "gpt2"          # GPT-2 
]

@st.cache_data
def get_tiktoken_encoding(encoding_name: str):
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        st.error(f"Error loading encoding {encoding_name}: {str(e)}")

def tokenize_text(text: str, encoding_name: str) -> Tuple[List[int], List[str], int]:
    if not text.strip():
        return [], [], 0

    encoding = get_tiktoken_encoding(encoding_name)
    if encoding is None:
        return [], [], 0
    try:
        token_ids = encoding.encode(text)
        decoded_tokens = []
        for token_id in token_ids:
            try:
                decoded_token = encoding.decode([token_id])
                decoded_tokens.append(decoded_token)
            except Exception:
                decoded_tokens.append(f"<ERROR_TOKEN_{token_id}>")
        return token_ids, decoded_tokens, len(token_ids)
    except Exception as e:
        st.error(f"Tokenization eror: {str(e)}")
        return [], [], 0

def calculate_costs(token_count: int) -> Dict[str, Dict[str, float]]:
    costs = {}
    for model, pricing in LLM_PRICING.items():
        input_cost = (token_count / 1000) * pricing["input"]
        output_cost = (token_count / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        costs[model] = {
            "input": input_cost,
            "output": output_cost,
            "total": total_cost
        }
    return costs

def create_token_length_distribution(decoded_tokens: List[str]) -> go.Figure:
    if not decoded_tokens:
        return go.Figure()

    token_lengths = [len(token) for token in decoded_tokens]
    length_counts = Counter(token_lengths)
    fig = go.Figure(data = [
        go.Bar(
            x = list(length_counts.keys()),
            y = list(length_counts.values()),
            marker_color = 'rgb(55, 83, 109)' #Lighter Blue
        )
    ])

    fig.update_layout(
        title = "Token Length Distribution",
        xaxis_title = "Token Length (chars)",
        yaxis_title = "Count",
        showlegend = False,
        height = 400
    )
    return fig

def create_token_breakdown_table(token_ids: List[int], decoded_tokens: List[str]) -> pd.DataFrame:
    if not token_ids or decoded_tokens:
        return pd.DataFrame()

    df = pd.DataFrame({
        'Position': range(1, len(token_ids) + 1),
        'Token ID': token_ids,
        'Decoded Token': [repr(token) for token in decoded_tokens],
        'Length': [len(token) for token in decoded_tokens],
        'Type': ['Whitespace' if token.isspace() else 'Alphanumeric' if token.isalnum() else 'Special' for token in decoded_tokens]
    })
    return df

def render_sidebar():
    st.sidebar.header("Configuration")
    selected_encoding = st.sidebar.selectbox(
        "Select Tokenizer Encoding:",
        TIKTOKEN_ENCODINGS,
        index = 0,
        help = "Choose the tokenization encoding"
    )
    st.sidebar.header("Display Options")
    show_token_table = st.sidebar.checkbox("Show Token Breakdown Table", value = True)
    show_length_dist = st.sidebar.checkbox("Show Length Distribution", value = True)
    show_cost_estimates = st.sidebar.checkbox("Show Cost Estimates", value = True)

    st.sidebar.header("Advanced Options")
    max_token_display = st.sidebar.slider(
        "Max Tokens in Table:",
        min_value = 10,
        max_value = 1000,
        value = 100,
        help = "Limit the number of tokens displayed in the breakdown table"
    )

    return selected_encoding, show_token_table, show_length_dist, show_cost_estimates, max_token_display

def render_cost_estimates(costs: Dict[str, Dict[str, float]], token_count: int):
    st.subheader("Cost Estimates")
    cost_data = []
    for model, pricing in costs.items():
        cost_data.append({
            'Model': model,
            'Input Cost ($)': f"${pricing['input']:.6f}",
            'Output Cost ($)': f"${pricing['output']:.6f}",
            'Total Cost ($)': f"${pricing['total']:.6f}"
        })
    cost_df = pd.DataFrame(cost_data)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(cost_df, use_container_width = True, hide_index = True)
    
    with col2:
        st.metric("Total Tokens", token_count)
        if token_count > 0:
            cheapest_model = min(costs.keys(), key=lambda x: costs[x]['total'])
            most_expensive_model = max(costs.keys(), key=lambda x: costs[x]['total'])
            
            st.metric(
                "Cheapest Model", 
                cheapest_model,
                f"${costs[cheapest_model]['total']:.6f}"
            )
            st.metric(
                "Most Expensive", 
                most_expensive_model,
                f"${costs[most_expensive_model]['total']:.6f}"
            )
def main():
    """Main application function."""
    
    # Header
    st.title("Token Usage Visualizer")
    st.markdown("""
    Analyze text tokenization and estimate costs across different Language Models.
    Enter your text below to see detailed tokenization insights.
    """)
    selected_encoding, show_token_table, show_length_dist, show_cost_estimates, max_tokens_display = render_sidebar()
    st.header("📝 Text Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "File Upload"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "Direct Text Input":
        text_input = st.text_area(
            "Enter your text or prompt:",
            height = 200,
            placeholder = "Type or paste your text here...",
            help = "Enter the text you want to analyze for tokenization"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt', 'md', 'py', 'json', 'csv'],
            help="Upload a text file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                text_input = str(uploaded_file.read(), "utf-8")
                st.success(f"File uploaded successfully! ({len(text_input)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Process text if available
    if text_input.strip():
        
        # Show processing indicator
        with st.spinner("Tokenizing text..."):
            start_time = time.time()
            token_ids, decoded_tokens, token_count = tokenize_text(text_input, selected_encoding)
            processing_time = time.time() - start_time
        
        if token_count > 0:
            st.header("📊 Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tokens", token_count)
            
            with col2:
                st.metric("Characters", len(text_input))
            
            with col3:
                char_per_token = len(text_input) / token_count if token_count > 0 else 0
                st.metric("Chars/Token", f"{char_per_token:.2f}")
            
            with col4:
                st.metric("Processing Time", f"{processing_time:.3f}s")
            
            # Cost estimates
            if show_cost_estimates:
                costs = calculate_costs(token_count)
                render_cost_estimates(costs, token_count)
            
            # Token breakdown table
            if show_token_table and decoded_tokens:
                st.header("🔍 Token Breakdown")
                
                # Limit tokens for display performance
                display_tokens = min(len(decoded_tokens), max_tokens_display)
                if display_tokens < len(decoded_tokens):
                    st.info(f"Showing first {display_tokens} tokens out of {len(decoded_tokens)} total tokens.")
                
                token_df = create_token_breakdown_table(
                    token_ids[:display_tokens], 
                    decoded_tokens[:display_tokens]
                )
                
                if not token_df.empty:
                    st.dataframe(
                        token_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    if len(decoded_tokens) > display_tokens:
                        full_token_df = create_token_breakdown_table(token_ids, decoded_tokens)
                        csv = full_token_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Token Breakdown (CSV)",
                            data=csv,
                            file_name="token_breakdown.csv",
                            mime="text/csv"
                        )
            if show_length_dist and decoded_tokens:
                st.header("📈 Token Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = create_token_length_distribution(decoded_tokens)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Token statistics
                    token_lengths = [len(token) for token in decoded_tokens]
                    
                    st.subheader("Statistics")
                    st.metric("Unique Tokens", len(set(decoded_tokens)))
                    st.metric("Avg Token Length", f"{sum(token_lengths)/len(token_lengths):.2f}")
                    st.metric("Max Token Length", max(token_lengths))
                    st.metric("Min Token Length", min(token_lengths))
                    token_types = Counter()
                    for token in decoded_tokens:
                        if token.isspace():
                            token_types['Whitespace'] += 1
                        elif token.isalnum():
                            token_types['Alphanumeric'] += 1
                        else:
                            token_types['Special'] += 1
                    
                    st.subheader("Token Types")
                    for token_type, count in token_types.items():
                        percentage = (count / len(decoded_tokens)) * 100
                        st.metric(token_type, f"{count} ({percentage:.1f}%)")
        
        else:
            st.warning("No tokens were generated. Please check your input text.")
    
    else:
        pass

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Tokenization by OpenAI's tiktoken
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()