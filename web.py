import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from collections import Counter
import spacy
from spacy.tokens import Doc, Span
import os
import nltk
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import base64

# Page configuration
st.set_page_config(
    page_title="NER Analysis Tool",
    page_icon="🔍",
    layout="wide",
)

# Initialize directories
if not os.path.exists("models"):
    os.makedirs("models")

# Set up sidebar
st.sidebar.title("NER Analysis")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Data Explorer", "Model Training", "Model Evaluation", "Interactive Annotation"],
)


# Utility functions (from your notebook)
@st.cache_data
def load_ner_data_from_txt(file_path, sep="\t", encoding="utf-8"):
    """Load NER data from a text file into a pandas DataFrame."""
    df = pd.read_csv(
        file_path,
        sep=sep,
        skip_blank_lines=False,
        encoding=encoding,
        names=["word", "tag"],
        header=None,
        quoting=3,
    )
    return df


def structure_data(df):
    """Add sentence IDs and other necessary columns."""
    df["sentence"] = df.isnull().all(axis=1).cumsum()
    df["pos"] = ""
    return df[["word", "pos", "tag", "sentence"]]


class SentenceGetter:
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False

        def agg_func(s):
            words = s["word"].values.tolist()
            pos = s["pos"].values.tolist()
            tags = s["tag"].values.tolist()
            return [(w, p, t) for w, p, t in zip(words, pos, tags)]

        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

def color_weight(val):
    if pd.isna(val):
        return ""
    if val > 0:
        intensity = min(abs(val) * 3, 1)
        return f"background-color: rgba(69, 117, 180, {intensity}); color: black;"
    else:
        intensity = min(abs(val) * 3, 1)
        return f"background-color: rgba(215, 48, 39, {intensity}); color: black;"
    
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    if isinstance(word, float) or word is None:
        word = ""
    if isinstance(postag, float) or postag is None:
        postag = ""

    features = [
        "bias",
        "word.lower=" + word.lower(),
        "word[-3:]=" + word[-3:] if len(word) >= 3 else "word[-3:]=",
        "word[-2:]=" + word[-2:] if len(word) >= 2 else "word[-2:]=",
        "word.isupper=%s" % word.isupper(),
        "word.istitle=%s" % word.istitle(),
        "word.isdigit=%s" % word.isdigit(),
        "postag=" + postag,
        "postag[:2]=" + postag[:2] if len(postag) >= 2 else "postag[:2]=",
    ]

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]

        # Handle NaN or None values for previous word
        if isinstance(word1, float) or word1 is None:
            word1 = ""
        if isinstance(postag1, float) or postag1 is None:
            postag1 = ""

        features.extend(
            [
                "-1:word.lower=" + word1.lower(),
                "-1:word.istitle=%s" % word1.istitle(),
                "-1:word.isupper=%s" % word1.isupper(),
                "-1:postag=" + postag1,
                "-1:postag[:2]=" + postag1[:2]
                if len(postag1) >= 2
                else "-1:postag[:2]=",
            ]
        )
    else:
        features.append("BOS")

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]

        # Handle NaN or None values for next word
        if isinstance(word1, float) or word1 is None:
            word1 = ""
        if isinstance(postag1, float) or postag1 is None:
            postag1 = ""

        features.extend(
            [
                "+1:word.lower=" + word1.lower(),
                "+1:word.istitle=%s" % word1.istitle(),
                "+1:word.isupper=%s" % word1.isupper(),
                "+1:postag=" + postag1,
                "+1:postag[:2]=" + postag1[:2]
                if len(postag1) >= 2
                else "+1:postag[:2]=",
            ]
        )
    else:
        features.append("EOS")

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [str(label) if not pd.isna(label) else "O" for _, _, label in sent]


def sent2tokens(sent):
    tokens = []
    for token, _, _ in sent:
        if isinstance(token, float) and pd.isna(token):
            tokens.append("")
        elif token is None:
            tokens.append("")
        else:
            tokens.append(str(token))
    return tokens


def visualize_ner_blocks(tokens, tags, title="NER Tags"):
    """
    Visualize NER tags with improved color contrast and grouping for same entities.
    """
    processed_tokens = []
    for token in tokens:
        if token == "":
            processed_tokens.append("[EMPTY]")
        else:
            processed_tokens.append(token)

    # Create a color map for entities with better contrast
    entity_colors = {}
    text_colors = {}
    unique_prefixes = set()

    for tag in tags:
        if tag != "O":
            prefix = tag.split("-")[1] if "-" in tag else tag
            unique_prefixes.add(prefix)

    # Color palette with better readability
    color_palette = [
        {"bg": "#E6F3FF", "text": "#003366"},  # Light blue bg, dark blue text
        {"bg": "#FFE6E6", "text": "#660000"},  # Light red bg, dark red text
        {"bg": "#E6FFE6", "text": "#006600"},  # Light green bg, dark green text
        {"bg": "#F2E6FF", "text": "#330066"},  # Light purple bg, dark purple text
        {"bg": "#FFF2E6", "text": "#663300"},  # Light orange bg, dark orange text
        {"bg": "#E6FFFF", "text": "#006666"},  # Light cyan bg, dark cyan text
        {"bg": "#FFE6F3", "text": "#660033"},  # Light pink bg, dark pink text
        {"bg": "#F3FFE6", "text": "#336600"},  # Light lime bg, dark lime text
    ]

    # Assign colors to entity types
    for i, prefix in enumerate(sorted(unique_prefixes)):
        color_idx = i % len(color_palette)
        entity_colors[prefix] = color_palette[color_idx]["bg"]
        text_colors[prefix] = color_palette[color_idx]["text"]

    # Build HTML for the token blocks with grouping
    blocks_html = ""
    i = 0
    while i < len(processed_tokens):
        current_tag = tags[i]

        if current_tag == "O":
            # Non-entity token
            blocks_html += f"""
            <div class="token">
                <div class="word">{processed_tokens[i]}</div>
                <div class="tag">O</div>
            </div>
            """
            i += 1
        else:
            # Begin entity group
            # Extract entity type
            prefix = current_tag.split("-")[1] if "-" in current_tag else current_tag
            tag_type = current_tag.split("-")[0] if "-" in current_tag else ""

            # Find all consecutive tokens with same entity type
            entity_tokens = [processed_tokens[i]]
            entity_indices = [i]
            j = i + 1
            while j < len(processed_tokens) and j < len(tags):
                next_tag = tags[j]
                if (
                    next_tag != "O"
                    and (next_tag.split("-")[1] if "-" in next_tag else next_tag)
                    == prefix
                ):
                    entity_tokens.append(processed_tokens[j])
                    entity_indices.append(j)
                    j += 1
                else:
                    break

            # Create grouped entity box
            entity_text = " ".join(entity_tokens)
            blocks_html += f"""
            <div class="token entity" style="background-color: {entity_colors[prefix]}; color: {text_colors[prefix]}; border-color: {text_colors[prefix]};">
                <div class="word">{entity_text}</div>
                <div class="tag">{current_tag} → {tags[j - 1]}</div>
                <div class="entity-type">{prefix}</div>
            </div>
            """
            i = j  # Skip to the end of this entity group

    # Create the complete HTML with CSS styling
    html = f"""
    <div style="margin-bottom: 20px;">
        <h3 style="margin-bottom: 10px;">{title}</h3>
        <style>
            .ner-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                font-family: sans-serif;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }}
            .token {{
                padding: 8px;
                border-radius: 4px;
                background-color: #f5f5f5;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-width: 50px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .entity {{
                border: 2px solid;
            }}
            .word {{
                font-weight: bold;
                margin-bottom: 4px;
                color: black;
            }}
            .tag {{
                font-size: 0.8em;
                color: #666;
            }}
            .entity-type {{
                font-size: 0.7em;
                font-weight: bold;
                margin-top: 2px;
            }}
        </style>
        <div class="ner-container">
            {blocks_html}
        </div>
    </div>
    """

    return html


def plot_confusion_matrix(
    y_true, y_pred, labels=None, normalize=False, title="Confusion Matrix"
):
    """Generate and plot a confusion matrix for NER tags."""
    y_true_flat = list(chain.from_iterable(y_true))
    y_pred_flat = list(chain.from_iterable(y_pred))

    if labels is None:
        labels = sorted(set(y_true_flat) | set(y_pred_flat))

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    return fig

def plot_entity_level_confusion_matrix(
    y_true,
    y_pred,
    figsize=(10, 8),
    normalize=False,
    title="Entity Type Confusion Matrix",
):
    """
    Generate and plot a confusion matrix at the entity type level
    (ignoring B-, I-, etc. prefixes).

    Parameters:
    -----------
    y_true : list
        List of true label sequences
    y_pred : list
        List of predicted label sequences
    figsize : tuple, default=(10, 8)
        Figure size for the plot
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    title : str
        Title for the confusion matrix plot
    """

    # Extract entity types (removing B-, I-, etc. prefixes)
    def extract_entity_types(tags):
        entity_types = []
        for tag in tags:
            if tag == "O":
                entity_types.append("O")
            else:
                # Split by hyphen and take the second part (the entity type)
                parts = tag.split("-", 1)
                if len(parts) > 1:
                    entity_types.append(parts[1])
                else:
                    entity_types.append(tag)
        return entity_types

    # Flatten and convert to entity types
    y_true_flat = list(chain.from_iterable(y_true))
    y_pred_flat = list(chain.from_iterable(y_pred))

    y_true_types = extract_entity_types(y_true_flat)
    y_pred_types = extract_entity_types(y_pred_flat)

    # Get unique entity types
    unique_types = sorted(set(y_true_types) | set(y_pred_types))

    # Create confusion matrix
    cm = confusion_matrix(y_true_types, y_pred_types, labels=unique_types)

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=unique_types,
        yticklabels=unique_types,
        ax=ax,
    )
    plt.title(title)
    plt.ylabel("True Entity Type")
    plt.xlabel("Predicted Entity Type")
    plt.tight_layout()

    return fig


# App pages
if page == "Data Explorer":
    st.title("Data Explorer")

    uploaded_file = st.file_uploader(
        "Upload NER Data (TSV format)",
        type=["txt", "tsv"],
        key="data_explorer_uploader",  # Add this line
    )

    if uploaded_file is not None:
        # Check if the file has changed from previous upload
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            # Clear any previous processing state
            if 'structured_df' in st.session_state:
                del st.session_state.structured_df
            st.session_state.pos_processed = False
            
            # Save the file temporarily with a unique name based on file name
            temp_file_path = f"temp_data_{st.session_state.current_file}.txt"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Store the path for future reference
            st.session_state.temp_file_path = temp_file_path
        
        # Load the data from the current temp file
        df = load_ner_data_from_txt(st.session_state.temp_file_path)
        df["sentence"] = df.isnull().all(axis=1).cumsum()

        # Show basic info
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tokens", df.shape[0])
        with col2:
            st.metric("Sentences", df["sentence"].max() + 1)
        with col3:
            entity_count = df["tag"].nunique() - (1 if df["tag"].isna().any() else 0)
            st.metric("Entity Types", entity_count)

        # Show tag distribution
        st.subheader("Entity Tag Distribution")
        # Filter out NaN values for the chart
        tag_counts = df["tag"].dropna().value_counts()
        if not tag_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            tag_counts.plot(kind="bar", ax=ax)
            st.pyplot(fig)

        # POS Tagging Option
        st.subheader("POS Tagging with SpaCy Nusantara")
        
        # Store processed dataframe in session state to avoid reprocessing
        if 'structured_df' not in st.session_state:
            st.session_state.structured_df = df.copy()
            st.session_state.structured_df["pos"] = ""  # Initialize pos column
            st.session_state.pos_processed = False

        use_pos_tagging = st.checkbox(
            "Add POS tags using SpaCy id_nusantara model", value=True
        )

        # Create structured dataframe
        structured_df = st.session_state.structured_df

        # Add POS tags if requested and not already processed
        if use_pos_tagging and not st.session_state.pos_processed:
            process_button = st.button("Process POS Tags")
            
            if process_button:
                with st.spinner("Processing POS tags with SpaCy id_nusantara..."):
                    # Try loading the model
                    try:
                        nlp = spacy.load("id_nusantara")
                        st.success("Successfully loaded id_nusantara model")
                    except:
                        st.warning(
                            "Downloading id_nusantara model (this might take a while the first time)..."
                        )
                        try:
                            spacy.cli.download("id_nusantara")
                            nlp = spacy.load("id_nusantara")
                            st.success(
                                "Successfully downloaded and loaded id_nusantara model"
                            )
                        except Exception as e:
                            st.error(f"Error loading SpaCy model: {e}")
                            st.warning("Proceeding without POS tagging")
                            use_pos_tagging = False

                    if use_pos_tagging:
                        # Initialize stats tracking
                        total_tokens = 0
                        matched_tokens = 0

                        # Process each sentence group
                        progress_bar = st.progress(0)
                        sentence_groups = structured_df.groupby("sentence")
                        total_sentences = len(sentence_groups)

                        for i, (sentence_id, group) in enumerate(sentence_groups):
                            # Skip empty rows
                            if group.isnull().all(axis=1).any():
                                continue

                            # Get words for this sentence
                            words = group["word"].fillna("").tolist()
                            sentence_text = " ".join(words)

                            # Skip empty sentences
                            if not sentence_text.strip():
                                continue

                            # Process with spaCy
                            doc = nlp(sentence_text)

                            # Match tokens back to DataFrame rows
                            token_index = 0
                            for j, row_idx in enumerate(group.index):
                                if j < len(doc) and token_index < len(doc):
                                    # Try to match tokens with words
                                    if (
                                        str(doc[token_index]).lower()
                                        == str(structured_df.loc[row_idx, "word"]).lower()
                                    ):
                                        structured_df.loc[row_idx, "pos"] = doc[
                                            token_index
                                        ].pos_
                                        token_index += 1
                                        matched_tokens += 1
                                    else:
                                        # For words that don't align perfectly
                                        structured_df.loc[row_idx, "pos"] = (
                                            doc[j].pos_ if j < len(doc) else ""
                                        )
                                else:
                                    structured_df.loc[row_idx, "pos"] = ""

                                total_tokens += 1

                            # Update progress bar
                            progress_bar.progress((i + 1) / total_sentences)

                        # Update session state
                        st.session_state.structured_df = structured_df
                        st.session_state.pos_processed = True
                        
                        # Summary statistics
                        match_rate = (
                            matched_tokens / total_tokens if total_tokens > 0 else 0
                        )
                        st.metric("POS Tagging Match Rate", f"{match_rate:.2%}")

                        # Show POS tag distribution
                        pos_counts = structured_df["pos"].value_counts()
                        if not pos_counts.empty:
                            st.subheader("POS Tag Distribution")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            pos_counts.head(10).plot(kind="bar", ax=ax)
                            plt.title("Top 10 POS Tags")
                            st.pyplot(fig)
                st.rerun()

        # Show sample data
        structured_df = st.session_state.structured_df[["word", "pos", "tag", "sentence"]]
        st.subheader("Sample Data")
        st.dataframe(structured_df.head(20))

        # Data type selection and save option
        if st.session_state.pos_processed or structured_df["pos"].str.strip().any():
            with st.form("save_data_form"):
                st.subheader("Save Processed Data")

                # Add dataset type selection
                dataset_type = st.radio(
                    "Select dataset type",
                    ["train", "test"],
                    help="Specify whether this data will be used for training or testing",
                )

                # Generate suggested filename based on selection
                default_filename = f"{dataset_type}_data_with_pos.csv"
                save_filename = st.text_input(
                    "Filename to save POS-tagged data", default_filename
                )

                # Form submit button
                submit_button = st.form_submit_button(
                    "Save Data for Model Training/Evaluation"
                )

                if submit_button:
                    structured_df.to_csv(save_filename, index=False)
                    st.success(f"Data saved as {save_filename}")

                    # Create a download link
                    csv = structured_df.to_csv(index=False).encode()
                    b64 = base64.b64encode(csv).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{save_filename}">Download CSV file</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Display information about next steps
                    if dataset_type == "train":
                        st.info(
                            "You can now use this data file in the 'Model Training' section."
                        )
                    else:
                        st.info(
                            "You can now use this data file in the 'Model Evaluation' section."
                        )

        # Function to limit tokens to 100 words
        def limit_tokens(tokens, tags, pos_tags=None, limit=100):
            if len(tokens) > limit:
                if pos_tags:
                    return tokens[:limit], tags[:limit], pos_tags[:limit]
                return tokens[:limit], tags[:limit]
            if pos_tags:
                return tokens, tags, pos_tags
            return tokens, tags

        # Show sentences (limited to 100 words)
        st.subheader("Sample Sentences (Limited to 100 words)")
        getter = SentenceGetter(structured_df)

        # Select random sentences to display
        import random

        num_samples = min(5, len(getter.sentences))
        sample_indices = random.sample(range(len(getter.sentences)), num_samples)

        for idx in sample_indices:
            sent = getter.sentences[idx]
            tokens = sent2tokens(sent)
            tags = sent2labels(sent)
            pos_tags = [pos for _, pos, _ in sent]

            # Limit tokens and tags to 100 words
            if len(tokens) > 100:
                tokens, tags, pos_tags = limit_tokens(tokens, tags, pos_tags, 100)

            st.markdown(f"**Sentence {idx}:** {' '.join(tokens)}")

            # Show POS tags if available
            if st.session_state.pos_processed:
                pos_display = ", ".join(
                    [f"{t}({p})" for t, p in zip(tokens, pos_tags) if p]
                )
                st.markdown(f"**POS Tags:** {pos_display}")

            html_output = visualize_ner_blocks(tokens, tags)
            st.html(html_output)

            if len(tokens) > 100:
                st.info(f"Showing first 100 words out of {len(tokens)} total words.")
            st.markdown("---")


# if page == "Data Explorer":
#     st.title("Data Explorer")
#     uploaded_file = st.file_uploader(
#         "Upload text file to analyze", 
#         type=["txt"],
#         key="interactive_annotation_uploader"  # Add this line
#     )

#     if uploaded_file is not None:
#         # Save the file temporarily
#         with open("temp_data.txt", "wb") as f:
#             f.write(uploaded_file.getvalue())

#         # Load the data
#         df = load_ner_data_from_txt("temp_data.txt")
#         df["sentence"] = df.isnull().all(axis=1).cumsum()

#         # Show basic info
#         st.subheader("Data Overview")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total Tokens", df.shape[0])
#         with col2:
#             st.metric("Sentences", df["sentence"].max() + 1)
#         with col3:
#             entity_count = df["tag"].nunique() - (1 if df["tag"].isna().any() else 0)
#             st.metric("Entity Types", entity_count)

#         # Show tag distribution
#         st.subheader("Entity Tag Distribution")
#         # Filter out NaN values for the chart
#         tag_counts = df["tag"].dropna().value_counts()
#         if not tag_counts.empty:
#             fig, ax = plt.subplots(figsize=(10, 6))
#             tag_counts.plot(kind="bar", ax=ax)
#             st.pyplot(fig)

#         # POS Tagging Option
#         st.subheader("POS Tagging with SpaCy Nusantara")
#         use_pos_tagging = st.checkbox(
#             "Add POS tags using SpaCy id_nusantara model", value=True
#         )

#         # Create structured dataframe
#         structured_df = df.copy()
#         structured_df["pos"] = ""  # Initialize pos column

#         # Add POS tags if requested
#         if use_pos_tagging:
#             with st.spinner("Processing POS tags with SpaCy id_nusantara..."):
#                 # Try loading the model
#                 try:
#                     nlp = spacy.load("id_nusantara")
#                     st.success("Successfully loaded id_nusantara model")
#                 except:
#                     st.warning("Downloading id_nusantara model (this might take a while the first time)...")
#                     try:
#                         spacy.cli.download("id_nusantara")
#                         nlp = spacy.load("id_nusantara")
#                         st.success("Successfully downloaded and loaded id_nusantara model")
#                     except Exception as e:
#                         st.error(f"Error loading SpaCy model: {e}")
#                         st.warning("Proceeding without POS tagging")
#                         use_pos_tagging = False
                
#                 if use_pos_tagging:
#                     # Initialize stats tracking
#                     total_tokens = 0
#                     matched_tokens = 0
                    
#                     # Process each sentence group
#                     progress_bar = st.progress(0)
#                     sentence_groups = structured_df.groupby("sentence")
#                     total_sentences = len(sentence_groups)
                    
#                     for i, (sentence_id, group) in enumerate(sentence_groups):
#                         # Skip empty rows
#                         if group.isnull().all(axis=1).any():
#                             continue
                            
#                         # Get words for this sentence
#                         words = group["word"].fillna("").tolist()
#                         sentence_text = " ".join(words)
                        
#                         # Skip empty sentences
#                         if not sentence_text.strip():
#                             continue
                            
#                         # Process with spaCy
#                         doc = nlp(sentence_text)
                        
#                         # Match tokens back to DataFrame rows
#                         token_index = 0
#                         for j, row_idx in enumerate(group.index):
#                             if j < len(doc) and token_index < len(doc):
#                                 # Try to match tokens with words
#                                 if (str(doc[token_index]).lower() == 
#                                     str(structured_df.loc[row_idx, "word"]).lower()):
#                                     structured_df.loc[row_idx, "pos"] = doc[token_index].pos_
#                                     token_index += 1
#                                     matched_tokens += 1
#                                 else:
#                                     # For words that don't align perfectly
#                                     structured_df.loc[row_idx, "pos"] = doc[j].pos_ if j < len(doc) else ""
#                             else:
#                                 structured_df.loc[row_idx, "pos"] = ""
                            
#                             total_tokens += 1
                        
#                         # Update progress bar
#                         progress_bar.progress((i + 1) / total_sentences)
                    
#                     # Summary statistics
#                     match_rate = matched_tokens / total_tokens if total_tokens > 0 else 0
#                     st.metric("POS Tagging Match Rate", f"{match_rate:.2%}")
                    
#                     # Show POS tag distribution
#                     pos_counts = structured_df["pos"].value_counts()
#                     if not pos_counts.empty:
#                         st.subheader("POS Tag Distribution")
#                         fig, ax = plt.subplots(figsize=(10, 6))
#                         pos_counts.head(10).plot(kind="bar", ax=ax)
#                         plt.title("Top 10 POS Tags")
#                         st.pyplot(fig)

#         # Show sample data
#         structured_df = structured_df[["word", "pos", "tag", "sentence"]]
#         st.subheader("Sample Data")
#         st.dataframe(structured_df.head(20))

#         # Data type selection and save option
#         if use_pos_tagging:
#             st.subheader("Save Processed Data")

#             # Add dataset type selection
#             dataset_type = st.radio(
#                 "Select dataset type",
#                 ["train", "test"],
#                 help="Specify whether this data will be used for training or testing",
#             )

#             # Generate suggested filename based on selection
#             default_filename = f"{dataset_type}_data_with_pos.csv"
#             save_filename = st.text_input("Filename to save POS-tagged data", default_filename)

#             if st.button("Save Data for Model Training/Evaluation"):
#                 structured_df.to_csv(save_filename, index=False)
#                 st.success(f"Data saved as {save_filename}")

#                 # Create a download link
#                 csv = structured_df.to_csv(index=False).encode()
#                 b64 = base64.b64encode(csv).decode()
#                 href = f'<a href="data:file/csv;base64,{b64}" download="{save_filename}">Download CSV file</a>'
#                 st.markdown(href, unsafe_allow_html=True)

#                 # Display information about next steps
#                 if dataset_type == "train":
#                     st.info("You can now use this data file in the 'Model Training' section.")
#                 else:
#                     st.info("You can now use this data file in the 'Model Evaluation' section.")

#         # Function to limit tokens to 100 words
#         def limit_tokens(tokens, tags, pos_tags=None, limit=100):
#             if len(tokens) > limit:
#                 if pos_tags:
#                     return tokens[:limit], tags[:limit], pos_tags[:limit]
#                 return tokens[:limit], tags[:limit]
#             if pos_tags:
#                 return tokens, tags, pos_tags
#             return tokens, tags

#         # Show sentences (limited to 100 words)
#         st.subheader("Sample Sentences (Limited to 100 words)")
#         getter = SentenceGetter(structured_df)

#         # Select random sentences to display
#         import random

#         num_samples = min(5, len(getter.sentences))
#         sample_indices = random.sample(range(len(getter.sentences)), num_samples)

#         for idx in sample_indices:
#             sent = getter.sentences[idx]
#             tokens = sent2tokens(sent)
#             tags = sent2labels(sent)
#             pos_tags = [pos for _, pos, _ in sent]

#             # Limit tokens and tags to 100 words
#             if len(tokens) > 100:
#                 tokens, tags, pos_tags = limit_tokens(tokens, tags, pos_tags, 100)

#             st.markdown(f"**Sentence {idx}:** {' '.join(tokens)}")

#             # Show POS tags if available
#             if use_pos_tagging:
#                 pos_display = ", ".join(
#                     [f"{t}({p})" for t, p in zip(tokens, pos_tags) if p]
#                 )
#                 st.markdown(f"**POS Tags:** {pos_display}")

#             html_output = visualize_ner_blocks(tokens, tags)
#             st.html(html_output)

#             if len(tokens) > 100:
#                 st.info(f"Showing first 100 words out of {len(tokens)} total words.")
#             st.markdown("---")
            
            

elif page == "Model Training":
    st.title("Model Training")

    # Initialize session state for training to prevent re-runs
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    # Initialize session state for loss tracking
    if "training_losses" not in st.session_state:
        st.session_state.training_losses = []
        st.session_state.validation_losses = []

    # Training data source options
    data_source = st.radio(
        "Select training data source",
        ["Upload new data file", "Use previously saved POS-tagged data"],
    )

    train_df = None

    if data_source == "Upload new data file":
        # Upload training data
        train_file = st.file_uploader(
            "Upload Training Data",
            type=["txt", "tsv", "csv"],
            key="model_training_uploader",
        )

        if train_file is not None:
            file_extension = train_file.name.split(".")[-1].lower()

            # Save the file temporarily
            with open(f"train_data.{file_extension}", "wb") as f:
                f.write(train_file.getvalue())

            if file_extension in ["txt", "tsv"]:
                # Load as TSV (standard NER format)
                train_df = load_ner_data_from_txt(f"train_data.{file_extension}")
                train_df = structure_data(train_df)
                st.success("Successfully loaded TSV/TXT format data")
            elif file_extension == "csv":
                # Load as CSV (likely from Data Explorer with POS tags)
                train_df = pd.read_csv(f"train_data.{file_extension}")
                required_columns = ["word", "pos", "tag", "sentence"]

                if all(col in train_df.columns for col in required_columns):
                    st.success("Successfully loaded CSV with POS tags")
                else:
                    st.error(
                        "CSV file is missing required columns. Please ensure it has: word, pos, tag, sentence"
                    )
                    train_df = None
    else:
        # List saved POS-tagged files
        saved_files = [
            f
            for f in os.listdir()
            if f.endswith("_with_pos.csv")
            or f.endswith("_pos.csv")
            or f.startswith("train_")
        ]

        if not saved_files:
            st.warning(
                "No saved POS-tagged files found. Please process data in Data Explorer first."
            )
        else:
            selected_file = st.selectbox("Select a saved file", saved_files)
            train_df = pd.read_csv(selected_file)
            st.success(f"Loaded data from {selected_file}")

    # If data is loaded, proceed with training
    if train_df is not None:
        # Show basic info
        st.subheader("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tokens", train_df.shape[0])
        with col2:
            sentence_count = len(train_df["sentence"].unique())
            st.metric("Sentences", sentence_count)
        with col3:
            entity_count = train_df["tag"].nunique() - (
                1 if train_df["tag"].isna().any() else 0
            )
            st.metric("Entity Types", entity_count)
        with col4:
            has_pos = train_df["pos"].str.strip().any()
            pos_status = "Yes" if has_pos else "No"
            st.metric("POS Tags", pos_status)

        # Show tag distribution
        tag_counts = train_df["tag"].dropna().value_counts()
        if not tag_counts.empty:
            st.subheader("Entity Tag Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            tag_counts.head(10).plot(kind="bar", ax=ax)
            plt.title("Top Entity Tags")
            st.pyplot(fig)

        # If we have POS tags, show their distribution
        if has_pos:
            pos_counts = train_df["pos"].dropna().value_counts()
            if not pos_counts.empty:
                st.subheader("POS Tag Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                pos_counts.head(10).plot(kind="bar", ax=ax)
                plt.title("Top 10 POS Tags")
                st.pyplot(fig)

        # Get sentences
        getter = SentenceGetter(train_df)
        sentences = getter.sentences

        # Training parameters
        st.subheader("Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            c1 = st.slider(
                "C1 (L1 regularization)", 0.0, 2.0, 0.1, 0.1
            )  # Lower default
            max_iterations = st.slider("Max Iterations", 10, 200, 50, 10)
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)

        with col2:
            c2 = st.slider(
                "C2 (L2 regularization)", 0.0, 0.1, 0.0001, 0.0001
            )  # Lower default
            model_name = st.text_input("Model Name", "ner_model")

        # Sample size
        max_sample = min(10000, len(sentences))
        sample_size = st.slider(
            "Training Sample Size", 100, max_sample, min(1000, max_sample), 100
        )

        # Split data into train and validation sets
        from sklearn.model_selection import train_test_split

        train_sentences, val_sentences = train_test_split(
            sentences[:sample_size], test_size=validation_split, random_state=42
        )

        st.info(
            f"Training on {len(train_sentences)} sentences, validating on {len(val_sentences)} sentences"
        )

        # Start training
        train_button = st.button("Train Model")

        if train_button:
            st.session_state.model_trained = True
            st.session_state.training_losses = []
            st.session_state.validation_losses = []

        if st.session_state.model_trained:
            with st.spinner(f"Training on {len(train_sentences)} sentences..."):
                # Prepare features and labels
                X_train = [sent2features(s) for s in train_sentences]
                y_train = [sent2labels(s) for s in train_sentences]

                # Prepare validation data
                X_val = [sent2features(s) for s in val_sentences]
                y_val = [sent2labels(s) for s in val_sentences]

                # Debug output
                st.write(f"Number of training instances: {len(X_train)}")
                st.write(f"Number of validation instances: {len(X_val)}")

                if not X_train or not y_train:
                    st.error("No valid training data. Please check your dataset.")
                else:
                    # Initialize trainer
                    trainer = pycrfsuite.Trainer(
                        verbose=True
                    )  # Set verbose to True for debugging

                    # Add training instances
                    for xseq, yseq in zip(X_train, y_train):
                        trainer.append(xseq, yseq)

                    # Set parameters
                    trainer.set_params(
                        {
                            "c1": c1,
                            "c2": c2,
                            "max_iterations": max_iterations,
                            "feature.possible_transitions": True,
                        }
                    )

                    # Progress display
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    loss_plot = st.empty()

                    # Calculate validation metric incrementally during training
                    # This uses a more reliable approach with temporary models
                    # for each evaluation point
                    def train_with_validation():
                        # Store the original trainer reference
                        original_trainer = trainer  # Save reference to the outer trainer
                        
                        for i in range(max_iterations):
                            # Train for a single iteration
                            temp_model_path = f"models/temp_{i}.crfsuite"
                            
                            # Set max_iterations to current iteration + 1
                            params = original_trainer.get_params()  # Use original_trainer
                            params["max_iterations"] = i + 1
                            original_trainer.set_params(params)  # Use original_trainer
                            
                            # Train model up to this iteration
                            original_trainer.train(temp_model_path)  # Use original_trainer
                            
                            # Get training loss for this iteration
                            if hasattr(original_trainer.logparser, 'iterations') and original_trainer.logparser.iterations:
                                current_loss = original_trainer.logparser.iterations[-1].get('loss', 0)
                                st.session_state.training_losses.append(current_loss)
                            
                            # Calculate validation error rate
                            temp_tagger = pycrfsuite.Tagger()
                            temp_tagger.open(temp_model_path)
                            
                            # Track validation metrics
                            val_errors = 0
                            val_total = 0
                            
                            # Evaluate on validation set
                            for x_seq, y_seq in zip(X_val, y_val):
                                y_pred = temp_tagger.tag(x_seq)
                                for y_true, y_p in zip(y_seq, y_pred):
                                    val_total += 1
                                    if y_true != y_p:
                                        val_errors += 1
                            
                            # Calculate validation error rate
                            val_loss = val_errors / val_total if val_total > 0 else 0
                            st.session_state.validation_losses.append(val_loss)
                            
                            # Update progress display
                            progress_bar.progress((i + 1) / max_iterations)
                            status_text.text(
                                f"Iteration {i+1}/{max_iterations}: "
                                f"Train Loss = {current_loss:.5f}, "
                                f"Val Loss = {val_loss:.5f}"
                            )
                            
                            # Plot the learning curves
                            fig, ax = plt.subplots(figsize=(10, 6))
                            iters = list(range(1, len(st.session_state.training_losses) + 1))
                            
                            ax.plot(iters, st.session_state.training_losses, 'b-', label='Training Loss')
                            ax.plot(iters, st.session_state.validation_losses, 'r-', label='Validation Loss')
                            
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Loss')
                            ax.set_title('Training and Validation Loss')
                            ax.legend()
                            ax.grid(True)
                            
                            loss_plot.pyplot(fig)
                            plt.close(fig)
                            
                            # Clean up temporary model file
                            if os.path.exists(temp_model_path):
                                os.remove(temp_model_path)
                            
                        # Final model training
                        model_path = f"models/{model_name}.crfsuite"
                        final_trainer = pycrfsuite.Trainer(verbose=False)  # Use a different name
                        
                        # Add all training instances again for final model
                        for xseq, yseq in zip(X_train, y_train):
                            final_trainer.append(xseq, yseq)
                        
                        # Set parameters for full training
                        final_trainer.set_params({
                            "c1": c1,
                            "c2": c2,
                            "max_iterations": max_iterations,
                            "feature.possible_transitions": True,
                        })
                        
                        # Train final model
                        final_trainer.train(model_path)
                        
                        return model_path

                    # Train with incremental validation
                    try:
                        final_model_path = train_with_validation()
                        st.success(
                            f"Model trained successfully and saved as {final_model_path}"
                        )

                        # Reset flag
                        st.session_state.model_trained = False

                        # Show final metrics
                        if (
                            st.session_state.training_losses
                            and st.session_state.validation_losses
                        ):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Final Training Loss",
                                    f"{st.session_state.training_losses[-1]:.5f}",
                                )
                            with col2:
                                st.metric(
                                    "Final Validation Loss",
                                    f"{st.session_state.validation_losses[-1]:.5f}",
                                )
                            with col3:
                                best_val_loss = min(st.session_state.validation_losses)
                                best_iter = (
                                    st.session_state.validation_losses.index(
                                        best_val_loss
                                    )
                                    + 1
                                )
                                st.metric(
                                    "Best Validation Loss",
                                    f"{best_val_loss:.5f} (Iter {best_iter})",
                                )

                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        st.session_state.model_trained = False
                    
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    # Add imports for model inspection
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics
    from itertools import chain
    import eli5

    # Model selection
    model_files = [f for f in os.listdir("models") if f.endswith(".crfsuite")]

    if not model_files:
        st.warning("No trained models found. Please train a model first.")
    else:
        selected_model = st.selectbox("Select a model", model_files)
        model_path = f"models/{selected_model}"

        # Add tabs for different evaluation tasks
        eval_task = st.radio(
            "Evaluation Task",
            ["Performance Evaluation", "Inspect Model Weights"],
            horizontal=True,
        )

        if eval_task == "Inspect Model Weights":
            st.subheader("Model Weight Inspection")

            # Try eli5 visualization first
            # Manual weight inspection (works with pycrfsuite models)
            with st.spinner("Loading model weights..."):
                try:
                    tagger = pycrfsuite.Tagger()
                    tagger.open(model_path)

                    # Get model information
                    info = tagger.info()

                    # Extract state and transition features
                    state_features = {i: w for i, w in info.state_features.items()}
                    transition_features = {i: w for i, w in info.transitions.items()}

                    # Basic model stats
                    st.write(f"**Model:** {selected_model}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("State Features", len(state_features))
                    with col2:
                        st.metric("Transition Features", len(transition_features))
                    with col3:
                        st.metric(
                            "Total Features",
                            len(state_features) + len(transition_features),
                        )

                    # Extract all entity types from model
                    entity_types = set()
                    for feat in state_features:
                        parts = feat[1].split(":")
                        if len(parts) > 1 and parts[0] != "O":
                            entity_type = (
                                parts[0].split("-")[1] if "-" in parts[0] else parts[0]
                            )
                            entity_types.add(entity_type)

                    # Filter options
                    st.subheader("Filter Features")

                    # Create filter options
                    filter_type = st.selectbox(
                        "Filter by entity type", ["All"] + sorted(list(entity_types))
                    )

                    feature_pattern = st.text_input(
                        "Filter by feature pattern (e.g., 'word.lower', 'postag')", ""
                    )

                    show_negative = st.checkbox("Show negative weights", value=True)

                    # Filter features
                    filtered_state_features = {}
                    for (attr, label), weight in state_features.items():
                        include = True

                        # Skip negative weights if not showing them
                        if not show_negative and weight < 0:
                            include = False

                        # Apply entity type filter
                        if filter_type != "All":
                            if "-" in label:
                                label_type = label.split("-")[1]
                            else:
                                label_type = label if label != "O" else ""
                            if label_type != filter_type:
                                include = False

                        # Apply feature pattern filter
                        if feature_pattern and feature_pattern not in attr:
                            include = False

                        if include:
                            filtered_state_features[(attr, label)] = weight

                    st.write(
                        f"**Showing {len(filtered_state_features)} of {len(state_features)} features**"
                    )

                    # Top feature weights
                    st.subheader("Top Feature Weights")

                    # Allow user to select how many features to show
                    top_n = st.slider("Number of top features to show", 5, 100, 30)

                    # Sort features by absolute weight
                    sorted_features = sorted(
                        filtered_state_features.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:top_n]

                    
                        # Replace the existing feature visualization code in your "Inspect Model Weights" section:

                    if sorted_features:
                        # Create a dataframe for better visualization with proper label handling
                        weights_data = []
                        for (attr, label), weight in sorted_features:
                            # Handle missing or empty labels
                            display_label = label if label and label.strip() else "O"
                            
                            # DON'T truncate the feature name here - keep it full
                            clean_attr = attr  # Keep full feature name
                            
                            weights_data.append({
                                "Feature": clean_attr,
                                "Label": display_label, 
                                "Weight": weight,
                                "Full_Feature": attr
                            })
                        
                        weights_df = pd.DataFrame(weights_data)

                        # Visualize weights with horizontal bar chart
                        fig, ax = plt.subplots(figsize=(18, max(10, top_n / 2)))  # Even bigger figure

                        # Prepare data for horizontal bar chart with FULL feature display
                        features_display = []
                        for _, row in weights_df.iterrows():
                            if row.Label == "O":
                                display_text = f"{row.Feature}"  # Show full feature for O tags
                            else:
                                display_text = f"{row.Feature} → {row.Label}"  # Show full feature → label
                            features_display.append(display_text)

                        weights = weights_df["Weight"].values

                        # Sort by weight value for better visualization
                        sorted_indices = np.argsort(weights)
                        features_display = [features_display[i] for i in sorted_indices]
                        weights = weights[sorted_indices]

                        # Color positive and negative weights differently
                        colors = ["#d73027" if w < 0 else "#4575b4" for w in weights]

                        # Create horizontal bar chart
                        bars = ax.barh(range(len(weights)), weights, color=colors, height=0.7)

                        # Add feature descriptions and values with better text handling
                        max_weight = max(abs(weights)) if len(weights) > 0 else 1
                        
                        # Set y-tick labels to show the feature names
                        ax.set_yticks(range(len(weights)))
                        
                        # Truncate only for y-tick labels if they're too long, but keep them readable
                        y_labels = []
                        for feature in features_display:
                            if len(feature) > 80:  # Increased limit
                                # Smart truncation - try to keep the important parts
                                if "→" in feature:
                                    parts = feature.split("→")
                                    feat_part = parts[0].strip()
                                    label_part = parts[1].strip()
                                    if len(feat_part) > 60:
                                        feat_part = feat_part[:57] + "..."
                                    y_labels.append(f"{feat_part} → {label_part}")
                                else:
                                    y_labels.append(feature[:77] + "...")
                            else:
                                y_labels.append(feature)
                        
                        ax.set_yticklabels(y_labels, fontsize=9)

                        # Add weight values on the bars
                        for i, (bar, weight) in enumerate(zip(bars, weights)):
                            # Position the text appropriately
                            if weight < 0:
                                text_x = weight - max_weight * 0.02
                                ha = "right"
                            else:
                                text_x = weight + max_weight * 0.02
                                ha = "left"

                            # Add weight value text
                            ax.text(
                                text_x,
                                i,
                                f"{weight:.4f}",  # Show 4 decimal places
                                ha=ha,
                                va="center",
                                color="black",
                                fontsize=8,
                                fontweight="bold"
                            )

                        # Add title and labels
                        ax.set_title(f"Top {top_n} Feature Weights by Magnitude", fontsize=16, fontweight='bold', pad=20)
                        ax.set_xlabel("Weight Value", fontsize=12, fontweight='bold')

                        # Add zero line
                        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.7, linewidth=1)

                        # Add grid
                        ax.grid(axis="x", linestyle="--", alpha=0.3)
                        ax.set_axisbelow(True)

                        # Adjust layout to prevent label cutoff
                        plt.tight_layout()
                        
                        # Add some padding to the x-axis limits for the text labels
                        x_min, x_max = ax.get_xlim()
                        x_range = x_max - x_min
                        ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)

                        st.pyplot(fig)
                        plt.close(fig)

                        # Show raw data in a table with better formatting - FULL FEATURES SHOWN
                        st.subheader("Feature Weights Table")
                        
                        # Create display dataframe with FULL feature names
                        display_df = weights_df[["Feature", "Label", "Weight"]].copy()

                        # Add weight colors
                        # def color_weight(val):
                        #     if pd.isna(val):
                        #         return ""
                        #     if val > 0:
                        #         intensity = min(abs(val) * 3, 1)
                        #         return f"background-color: rgba(69, 117, 180, {intensity}); color: black;"
                        #     else:
                        #         intensity = min(abs(val) * 3, 1)
                        #         return f"background-color: rgba(215, 48, 39, {intensity}); color: black;"

                        # Display styled table with full feature names
                        st.dataframe(
                            display_df.style
                                    .format({"Weight": "{:.6f}"})  # More decimal places
                                    .applymap(color_weight, subset=["Weight"])
                                    .set_properties(**{
                                        'font-size': '11px',
                                        'text-align': 'left'
                                    }),
                            use_container_width=True,
                            height=600  # Make table taller to show more features
                        )

                        # Export weights to CSV with FULL feature names
                        csv = weights_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Feature Weights",
                            data=csv,
                            file_name=f"{selected_model}_weights.csv",
                            mime="text/csv",
                        )
                        
                        # Add a search/filter feature for the table
                        st.subheader("Search Features")
                        search_term = st.text_input("Search for specific features (e.g., 'postag=NOUN', 'word.lower=jakarta')")
                        
                        if search_term:
                            filtered_df = display_df[display_df['Feature'].str.contains(search_term, case=False, na=False)]
                            if not filtered_df.empty:
                                st.write(f"Found {len(filtered_df)} features matching '{search_term}':")
                                st.dataframe(
                                    filtered_df.style
                                            .format({"Weight": "{:.6f}"})
                                            .applymap(color_weight, subset=["Weight"])
                                            .set_properties(**{
                                                'font-size': '11px',
                                                'text-align': 'left'
                                            }),
                                    use_container_width=True
                                )
                            else:
                                st.info(f"No features found matching '{search_term}'")
                    else:
                        st.warning("No features match your current filter settings.")
                        # Show raw data in a table with better formatting
                        st.subheader("Feature Weights Table")

                        # Create display dataframe without the Full_Feature column
                        display_df = weights_df[["Feature", "Label", "Weight"]].copy()

                        # Add weight colors
                        # def color_weight(val):
                        #     if pd.isna(val):
                        #         return ""
                        #     if val > 0:
                        #         intensity = min(abs(val) * 3, 1)
                        #         return f"background-color: rgba(69, 117, 180, {intensity}); color: black;"
                        #     else:
                        #         intensity = min(abs(val) * 3, 1)
                        #         return f"background-color: rgba(215, 48, 39, {intensity}); color: black;"

                        # Display styled table
                        st.dataframe(
                            display_df.style.format({"Weight": "{:.4f}"}).applymap(
                                color_weight, subset=["Weight"]
                            )
                        )

                        # Export weights to CSV
                        csv = weights_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Feature Weights",
                            data=csv,
                            file_name=f"{selected_model}_weights.csv",
                            mime="text/csv",
                        )
                    

                    # Show transition weights
                    if st.checkbox("Show transition weights", value=False):
                        st.subheader("Transition Weights")

                        # Sort transitions by absolute weight
                        sorted_transitions = sorted(
                            transition_features.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )

                        # Limit to top transitions
                        top_transitions = sorted_transitions[
                            : min(50, len(sorted_transitions))
                        ]

                        trans_df = pd.DataFrame(
                            [(f[0][0], f[0][1], f[1]) for f in top_transitions],
                            columns=["From", "To", "Weight"],
                        )

                        # Display transition weights as table
                        st.write(
                            "Top transition weights (probability of tag transitions)"
                        )

                        

                        st.dataframe(
                            trans_df.style.format({"Weight": "{:.4f}"}).applymap(
                                color_weight, subset=["Weight"]
                            )
                        )

                        # Create transition matrix visualization
                        # if not trans_df.empty:
                        #     # Get unique labels
                        #     all_labels = sorted(
                        #         set(trans_df["From"].unique())
                        #         | set(trans_df["To"].unique())
                        #     )

                        #     # Create transition matrix
                        #     trans_matrix = np.zeros((len(all_labels), len(all_labels)))
                        #     label_to_idx = {
                        #         label: i for i, label in enumerate(all_labels)
                        #     }

                        #     # Fill in transition matrix
                        #     for _, row in trans_df.iterrows():
                        #         from_idx = label_to_idx[row["From"]]
                        #         to_idx = label_to_idx[row["To"]]
                        #         trans_matrix[from_idx, to_idx] = row["Weight"]

                            # # Visualize as heatmap
                            # fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure

                            # # Create a more readable heatmap
                            # sns.heatmap(
                            #     trans_matrix,
                            #     annot=True,
                            #     fmt=".3f",  # Show 3 decimal places
                            #     cmap="RdBu_r",  # Better color scheme (red-blue reversed)
                            #     center=0,  # Center the colormap at 0
                            #     xticklabels=all_labels,
                            #     yticklabels=all_labels,
                            #     ax=ax,
                            #     cbar_kws={'label': 'Transition Weight'},  # Add colorbar label
                            #     square=True,  # Make cells square
                            #     linewidths=0.5,  # Add grid lines
                            #     annot_kws={'size': 10}  # Larger annotation text
                            # )

                            # # Improve title and labels
                            # plt.title("Tag Transition Weights Matrix", fontsize=16, fontweight='bold', pad=20)
                            # plt.xlabel("To (Next Tag)", fontsize=12, fontweight='bold')
                            # plt.ylabel("From (Current Tag)", fontsize=12, fontweight='bold')

                            # # Rotate labels for better readability
                            # plt.xticks(rotation=45, ha='right')
                            # plt.yticks(rotation=0)

                            # # Adjust layout
                            # plt.tight_layout()
                            # st.pyplot(fig)
                            # plt.close(fig)


                    # Feature importance by entity type
                    st.subheader("Feature Importance by Entity Type")

                    # Group features by entity type
                    entity_features = {}
                    for (attr, label), weight in state_features.items():
                        if label != "O" and "-" in label:
                            entity_type = label.split("-")[1]
                            if entity_type not in entity_features:
                                entity_features[entity_type] = []
                            entity_features[entity_type].append((attr, weight))

                    if entity_features:
                        # Display top features for each entity type
                        entity_type_to_show = st.selectbox(
                            "Select entity type to analyze",
                            sorted(entity_features.keys()),
                        )

                        if entity_type_to_show in entity_features:
                            # Sort features by absolute weight
                            sorted_entity_features = sorted(
                                entity_features[entity_type_to_show],
                                key=lambda x: abs(x[1]),
                                reverse=True,
                            )[:20]  # Show top 20 instead of 15

                            # Create dataframe
                            entity_df = pd.DataFrame(
                                [(f[0], f[1]) for f in sorted_entity_features],
                                columns=["Feature", "Weight"],
                            )

                            # Create much bigger bar chart
                            fig, ax = plt.subplots(figsize=(16, 12))  # Much larger figure size
                            colors = [
                                "#d73027" if w < 0 else "#4575b4"
                                for w in entity_df["Weight"]
                            ]

                            # Sort for better visualization
                            entity_df = entity_df.sort_values("Weight")

                            # Truncate long feature names for better display
                            display_features = []
                            for feature in entity_df["Feature"]:
                                if len(feature) > 60:
                                    display_features.append(feature[:57] + "...")
                                else:
                                    display_features.append(feature)

                            bars = ax.barh(
                                range(len(entity_df)), 
                                entity_df["Weight"], 
                                color=colors,
                                height=0.8  # Make bars thicker
                            )

                            # Set custom y-tick labels with truncated feature names
                            ax.set_yticks(range(len(entity_df)))
                            ax.set_yticklabels(display_features, fontsize=11)

                            # Add value labels on the bars
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                # Position label based on bar direction
                                if width >= 0:
                                    label_x = width + max(entity_df["Weight"]) * 0.01
                                    ha = 'left'
                                else:
                                    label_x = width - max(abs(entity_df["Weight"])) * 0.01
                                    ha = 'right'
                                
                                ax.text(
                                    label_x,
                                    bar.get_y() + bar.get_height() / 2,
                                    f'{width:.3f}',
                                    va='center',
                                    ha=ha,
                                    fontsize=10,
                                    fontweight='bold'
                                )

                            # Improve the plot appearance
                            plt.title(
                                f"Top {len(entity_df)} Features for Entity Type: {entity_type_to_show}",
                                fontsize=16,
                                fontweight='bold',
                                pad=20
                            )
                            plt.xlabel("Feature Weight", fontsize=14, fontweight='bold')
                            plt.ylabel("Features", fontsize=14, fontweight='bold')
                            
                            # Add grid for better readability
                            ax.grid(axis='x', linestyle='--', alpha=0.7)
                            ax.set_axisbelow(True)
                            
                            # Add a vertical line at x=0
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                            
                            # Adjust layout to prevent label cutoff
                            plt.tight_layout()
                            
                            # Add some padding to the x-axis limits
                            x_min, x_max = ax.get_xlim()
                            x_range = x_max - x_min
                            ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
                            
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Also show the raw data in an expandable section
                            with st.expander("View Raw Feature Data"):
                                # Create a more detailed dataframe
                                detailed_df = pd.DataFrame(
                                    [(f[0], f[1]) for f in sorted_entity_features],
                                    columns=["Feature", "Weight"]
                                )
                                
                                # Add color styling
                                def style_weights(val):
                                    if val > 0:
                                        return 'background-color: rgba(69, 117, 180, 0.3); color: black;'
                                    else:
                                        return 'background-color: rgba(215, 48, 39, 0.3); color: black;'
                                
                                st.dataframe(
                                    detailed_df.style
                                            .format({"Weight": "{:.4f}"})
                                            .applymap(style_weights, subset=["Weight"])
                                            .set_properties(**{'font-size': '12px'}),
                                    use_container_width=True
                                )

                    # Add explanation like eli5.show_weights would do
                    st.subheader("Model Weight Interpretation Guide")
                    st.markdown("""
                    **How to interpret these weights:**
                    
                    - **Positive weights (blue)**: Features that increase the likelihood of predicting a specific tag
                    - **Negative weights (red)**: Features that decrease the likelihood of predicting a specific tag
                    - **Feature format**: `feature=value → tag` means this feature-value pair influences the model to predict that tag
                    - **Transition weights**: Show how likely one tag is to follow another
                    
                    **Common feature prefixes:**
                    - `word.lower=` - Lowercase word form
                    - `postag=` - Part-of-speech tag
                    - `-1:` - Features from previous token
                    - `+1:` - Features from next token
                    - `word.isupper`, `word.istitle` - Capitalization features
                    
                    Higher absolute weight values indicate stronger influence on the model's decisions.
                    """)

                except Exception as e:
                    st.error(f"Error loading model weights: {e}")
                    st.info(
                        "The model may not support weight inspection or the format might be incompatible."
                    )

        else:  # Performance Evaluation
            # FIXED: Define data_source here
            data_source = st.radio(
                "Select test data source",
                ["Upload test data file", "Use previously saved POS-tagged data"],
            )

            test_df = None

            if data_source == "Upload test data file":
                # Test data upload
                test_file = st.file_uploader(
                    "Upload Test Data",
                    type=["txt", "tsv", "csv"],
                    key="model_evaluation_uploader",
                )
                if test_file is not None:
                    file_extension = test_file.name.split(".")[-1].lower()

                    # Save the file temporarily
                    with open(f"test_data.{file_extension}", "wb") as f:
                        f.write(test_file.getvalue())

                    if file_extension in ["txt", "tsv"]:
                        # Load as TSV (standard NER format)
                        test_df = load_ner_data_from_txt(f"test_data.{file_extension}")
                        test_df = structure_data(test_df)
                        st.success("Successfully loaded TSV/TXT format test data")
                    elif file_extension == "csv":
                        # Load as CSV (likely from Data Explorer with POS tags)
                        test_df = pd.read_csv(f"test_data.{file_extension}")
                        required_columns = ["word", "pos", "tag", "sentence"]

                        if all(col in test_df.columns for col in required_columns):
                            st.success("Successfully loaded CSV with POS tags")
                        else:
                            st.error(
                                "CSV file is missing required columns. Please ensure it has: word, pos, tag, sentence"
                            )
                            test_df = None
            else:
                # List saved POS-tagged files
                saved_files = [
                    f
                    for f in os.listdir()
                    if f.endswith("_with_pos.csv")
                    or f.endswith("_pos.csv")
                    or f.startswith("test_")
                ]

                if not saved_files:
                    st.warning(
                        "No saved POS-tagged files found. Please process data in Data Explorer first."
                    )
                else:
                    selected_file = st.selectbox(
                        "Select a saved file for testing", saved_files
                    )
                    test_df = pd.read_csv(selected_file)
                    st.success(f"Loaded test data from {selected_file}")

            # If data is loaded, proceed with evaluation
            if test_df is not None:
                # Show basic info
                st.subheader("Test Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tokens", test_df.shape[0])
                with col2:
                    sentence_count = len(test_df["sentence"].unique())
                    st.metric("Sentences", sentence_count)
                with col3:
                    entity_count = test_df["tag"].nunique() - (
                        1 if test_df["tag"].isna().any() else 0
                    )
                    st.metric("Entity Types", entity_count)
                with col4:
                    has_pos = test_df["pos"].str.strip().any()
                    pos_status = "Yes" if has_pos else "No"
                    st.metric("POS Tags", pos_status)

                # Show tag distribution
                tag_counts = test_df["tag"].dropna().value_counts()
                if not tag_counts.empty:
                    st.subheader("Entity Tag Distribution in Test Data")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    tag_counts.head(10).plot(kind="bar", ax=ax)
                    plt.title("Top Entity Tags in Test Data")
                    st.pyplot(fig)

                # If we have POS tags, show their distribution
                if has_pos:
                    pos_counts = test_df["pos"].dropna().value_counts()
                    if not pos_counts.empty:
                        st.subheader("POS Tag Distribution in Test Data")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pos_counts.head(10).plot(kind="bar", ax=ax)
                        plt.title("Top 10 POS Tags")
                        st.pyplot(fig)

                # Get sentences
                getter = SentenceGetter(test_df)
                test_sentences = getter.sentences

                # Limit test size for performance
                test_size = st.slider(
                    "Test Sample Size", 100, min(5000, len(test_sentences)), 500, 100
                )
                test_sentences = test_sentences[:test_size]

                # Evaluation scheme
                evaluation_scheme = st.selectbox(
                    "Evaluation Scheme",
                    ["Default", "IOB2", "BILOU"],
                    help="Default means : the tags are not converted to any scheme",
                )

                # Evaluate with sklearn-crfsuite option
                use_sklearn_crfsuite = st.checkbox(
                    "Use sklearn-crfsuite for evaluation",
                    value=True,
                    help="Enable additional metrics and evaluation capabilities",
                )

                # Evaluate button
                if st.button("Evaluate Model"):
                    with st.spinner(f"Evaluating on {test_size} sentences..."):
                        # Prepare features and labels
                        X_test = [sent2features(s) for s in test_sentences]
                        y_test = [sent2labels(s) for s in test_sentences]

                        # Load the model with pycrfsuite (standard)
                        tagger = pycrfsuite.Tagger()
                        tagger.open(model_path)

                        # Make predictions
                        y_pred = [tagger.tag(xseq) for xseq in X_test]

                        # For sklearn-crfsuite evaluation, extract labels from data
                        if use_sklearn_crfsuite:
                            try:
                                # Extract unique labels for metrics calculation
                                labels = list(set(chain.from_iterable(y_test)))
                                # Remove 'O' label for better focus on named entities
                                if "O" in labels:
                                    labels.remove("O")

                                # Add tabs for different evaluation views
                                eval_tabs = st.tabs(
                                    [
                                        "Basic Metrics",
                                        "Detailed Metrics",
                                        "Per-class Metrics",
                                        "Confusion Matrix",
                                    ]
                                )

                                with eval_tabs[0]:
                                    st.subheader("Basic Metrics")

                                    # Calculate metrics using sklearn-crfsuite
                                    flat_f1 = metrics.flat_f1_score(
                                        y_test,
                                        y_pred,
                                        average="weighted",
                                        labels=labels,
                                    )
                                    flat_accuracy = metrics.flat_accuracy_score(
                                        y_test, y_pred
                                    )

                                    # Display metrics in columns
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "Flat F1 Score (weighted)", f"{flat_f1:.4f}"
                                        )
                                    with col2:
                                        st.metric(
                                            "Flat Accuracy", f"{flat_accuracy:.4f}"
                                        )

                                    # Calculate sequence accuracy (% of sentences with all tags correct)
                                    sequence_correct = sum(
                                        1
                                        for y_t, y_p in zip(y_test, y_pred)
                                        if y_t == y_p
                                    )
                                    sequence_accuracy = (
                                        sequence_correct / len(y_test) if y_test else 0
                                    )
                                    st.metric(
                                        "Sequence Accuracy", f"{sequence_accuracy:.4f}"
                                    )

                                with eval_tabs[1]:
                                    st.subheader("Detailed Metrics")

                                    # Classification report based on selected scheme
                                    if evaluation_scheme == "IOB2":
                                        from seqeval.scheme import IOB2
                                        # Convert tags if needed
                                        report = seq_classification_report(
                                            y_test, y_pred, scheme=IOB2, digits=4
                                        )
                                        f1 = f1_score(
                                            y_test, y_pred, average="weighted", scheme=IOB2
                                        )
                                    elif evaluation_scheme == "BILOU":
                                        from seqeval.scheme import BILOU
                                        report = seq_classification_report(
                                            y_test, y_pred, scheme=BILOU, digits=4
                                        )
                                        f1 = f1_score(
                                            y_test, y_pred, average="weighted", scheme=BILOU
                                        )
                                    else:
                                        report = seq_classification_report(
                                            y_test, y_pred, digits=4, scheme=None
                                        )
                                        f1 = f1_score(
                                            y_test, y_pred, average="weighted", scheme=None
                                        )

                                    st.text(report)
                                    st.metric("Weighted F1 Score", f"{f1:.4f}")

                                    # Flat classification report from sklearn-crfsuite - FIXED
                                    st.subheader("Flat Token Classification Report")
                                    flat_report = metrics.flat_classification_report(
                                        y_test, y_pred, labels=labels
                                        # ,digits=4
                                    )
                                    st.text(flat_report)

                                with eval_tabs[2]:
                                    st.subheader("Per-class Metrics")

                                    # Calculate per-class precision, recall, F1 - FIXED
                                    sorted_labels = sorted(labels)
                                    class_metrics = metrics.flat_classification_report(
                                        y_test,
                                        y_pred,
                                        labels=sorted_labels,
                                        # output_dict=True,
                                        # digits=4,
                                    )

                                    # Filter out non-class entries (like 'accuracy', 'macro avg', etc.)
                                    class_df = pd.DataFrame({
                                        k: v for k, v in class_metrics.items() 
                                        if k in sorted_labels and isinstance(v, dict)
                                    }).T

                                    # Add color formatting
                                    def color_scale(val):
                                        if isinstance(val, (int, float)) and not pd.isna(val):
                                            if val > 0.9:
                                                return "background-color: #22543d; color: #e6ffe6;"
                                            elif val > 0.7:
                                                return "background-color: #2a4365; color: #e0eaff;"
                                            elif val > 0.5:
                                                return "background-color: #744210; color: #ffe6cc;"
                                            else:
                                                return "background-color: #742a2a; color: #ffeaea;"
                                        return ""

                                    # Display styled dataframe if we have data
                                    if not class_df.empty:
                                        st.dataframe(
                                            class_df.style.format("{:.4f}").applymap(
                                                color_scale,
                                                subset=["precision", "recall", "f1-score"],
                                            )
                                        )

                                        # F1 Score by class visualization
                                        if "f1-score" in class_df.columns:
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            class_df["f1-score"].sort_values().plot(kind="barh", ax=ax)
                                            plt.title("F1 Score by Entity Type")
                                            plt.xlabel("F1 Score")
                                            st.pyplot(fig)
                                            plt.close(fig)
                                    else:
                                        st.info("No per-class metrics available for the selected labels.")
                                with eval_tabs[3]:
                                    st.subheader("Confusion Matrix")

                                    # Entity-Level Confusion Matrix
                                    fig = plot_entity_level_confusion_matrix(
                                        y_test,
                                        y_pred,
                                        title="Entity Type Confusion Matrix",
                                    )
                                    st.pyplot(fig)

                                    # Show normalized confusion matrix
                                    fig = plot_entity_level_confusion_matrix(
                                        y_test,
                                        y_pred,
                                        normalize=True,
                                        title="Entity Type Confusion Matrix (Normalized)",
                                    )
                                    st.pyplot(fig)

                                    # Top misclassifications
                                    st.subheader("Top Misclassifications")

                                    y_true_flat = list(chain.from_iterable(y_test))
                                    y_pred_flat = list(chain.from_iterable(y_pred))

                                    # Create a DataFrame of misclassifications
                                    misclass_df = pd.DataFrame(
                                        {"True": y_true_flat, "Predicted": y_pred_flat}
                                    )
                                    misclass_df = misclass_df[
                                        misclass_df["True"] != misclass_df["Predicted"]
                                    ]

                                    # Count misclassifications
                                    misclass_counts = (
                                        misclass_df.groupby(["True", "Predicted"])
                                        .size()
                                        .reset_index(name="Count")
                                    )
                                    misclass_counts = misclass_counts.sort_values(
                                        "Count", ascending=False
                                    ).head(10)

                                    if not misclass_counts.empty:
                                        st.dataframe(misclass_counts)
                                    else:
                                        st.info(
                                            "No misclassifications found in the sample."
                                        )

                            except Exception as e:
                                st.error(
                                    f"Error during sklearn-crfsuite evaluation: {str(e)}"
                                )
                                st.info("Falling back to standard evaluation...")
                                use_sklearn_crfsuite = False

                        # Standard evaluation if sklearn-crfsuite is not used or failed
                        if not use_sklearn_crfsuite:
                            # Classification report based on selected scheme
                            st.subheader("Token-level Classification Report")

                            if evaluation_scheme == "IOB2":
                                from seqeval.scheme import IOB2

                                # Convert tags if needed
                                report = seq_classification_report(
                                    y_test, y_pred, scheme=IOB2, digits=4
                                )
                                f1 = f1_score(
                                    y_test, y_pred, average="weighted", scheme=IOB2
                                )
                            elif evaluation_scheme == "BILOU":
                                from seqeval.scheme import BILOU

                                report = seq_classification_report(
                                    y_test, y_pred, scheme=BILOU, digits=4
                                )
                                f1 = f1_score(
                                    y_test, y_pred, average="weighted", scheme=BILOU
                                )
                            else:
                                report = seq_classification_report(
                                    y_test, y_pred, digits=4, scheme=None
                                )
                                f1 = f1_score(
                                    y_test, y_pred, average="weighted", scheme=None
                                )

                            st.text(report)

                            # F1 score
                            st.metric("Weighted F1 Score", f"{f1:.4f}")

                            # Entity-Level Confusion Matrix
                            st.subheader("Entity-Level Confusion Matrix")
                            fig = plot_entity_level_confusion_matrix(
                                y_test, y_pred, title="Entity Type Confusion Matrix"
                            )
                            st.pyplot(fig)

                            # Show normalized confusion matrix
                            fig = plot_entity_level_confusion_matrix(
                                y_test,
                                y_pred,
                                normalize=True,
                                title="Entity Type Confusion Matrix (Normalized)",
                            )
                            st.pyplot(fig)

                        # Additional stats about model performance (for both evaluation methods)
                        if has_pos:
                            st.subheader("Performance Analysis by POS Tag")

                            # Flatten predictions and actual values
                            y_true_flat = []
                            y_pred_flat = []
                            pos_flat = []

                            for i, sent in enumerate(test_sentences):
                                for j, (token, pos, _) in enumerate(sent):
                                    if j < len(y_test[i]) and j < len(y_pred[i]):
                                        y_true_flat.append(y_test[i][j])
                                        y_pred_flat.append(y_pred[i][j])
                                        pos_flat.append(pos)

                            # Group by POS tag and calculate accuracy
                            pos_df = pd.DataFrame(
                                {
                                    "true": y_true_flat,
                                    "pred": y_pred_flat,
                                    "pos": pos_flat,
                                }
                            )

                            # Calculate accuracy per POS tag
                            pos_accuracy = (
                                pos_df.groupby("pos")
                                .apply(lambda x: (x["true"] == x["pred"]).mean())
                                .reset_index()
                            )
                            pos_accuracy.columns = ["POS Tag", "Accuracy"]
                            pos_accuracy = pos_accuracy.sort_values(
                                "Accuracy", ascending=False
                            )

                            # Display accuracy by POS tag
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(
                                x="POS Tag",
                                y="Accuracy",
                                data=pos_accuracy.head(10),
                                ax=ax,
                            )
                            plt.title("Model Accuracy by POS Tag (Top 10)")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)

                            # Show the full data
                            st.write("Accuracy by POS Tag:")
                            st.dataframe(pos_accuracy)

                        # Sample predictions
                        st.subheader("Sample Predictions")

                        # Select a few random test sentences
                        import random

                        sample_indices = random.sample(
                            range(len(test_sentences)), min(5, len(test_sentences))
                        )

                        for idx in sample_indices:
                            sent = test_sentences[idx]
                            tokens = sent2tokens(sent)
                            pos_tags = [pos for _, pos, _ in sent]
                            true_tags = sent2labels(sent)
                            pred_tags = tagger.tag(sent2features(sent))

                            st.markdown(f"**Text:** {' '.join(tokens)}")

                            # Show POS tags if available
                            if has_pos and any(pos.strip() for pos in pos_tags):
                                pos_display = ", ".join(
                                    [
                                        f"{t}({p})"
                                        for t, p in zip(tokens, pos_tags)
                                        if p.strip()
                                    ]
                                )
                                st.markdown(f"**POS Tags:** {pos_display}")

                            # Visualize true and predicted tags
                            st.html(
                                visualize_ner_blocks(
                                    tokens, true_tags, "True NER Tags"
                                ),
                            )
                            st.html(
                                visualize_ner_blocks(
                                    tokens, pred_tags, "Predicted NER Tags"
                                ),
                            )
                            st.markdown("---")

                        # Add export option for evaluation results
                        st.subheader("Export Evaluation Results")

                        # Create evaluation report DataFrame
                        eval_df = pd.DataFrame()

                        # Add predictions for each sentence
                        for i, (sentence, y_t, y_p) in enumerate(
                            zip(test_sentences, y_test, y_pred)
                        ):
                            tokens = sent2tokens(sentence)
                            for j, (token, true_tag, pred_tag) in enumerate(
                                zip(tokens, y_t, y_p)
                            ):
                                eval_df = eval_df.append(
                                    {
                                        "Sentence": i,
                                        "Token": token,
                                        "True_Tag": true_tag,
                                        "Predicted_Tag": pred_tag,
                                        "Correct": true_tag == pred_tag,
                                    },
                                    ignore_index=True,
                                )

                        # Convert to CSV for download
                        csv = eval_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Evaluation Results",
                            data=csv,
                            file_name="ner_evaluation_results.csv",
                            mime="text/csv",
                        )
                        
elif page == "Interactive Annotation":
    st.title("NER Text Analysis")

    # Model selection
    model_files = [f for f in os.listdir("models") if f.endswith(".crfsuite")]

    if not model_files:
        st.error(
            "No trained models found. Please train a model first in the Model Training section."
        )
    else:
        # Select a model for analysis
        selected_model = st.selectbox("Select a model", model_files)
        model_path = f"models/{selected_model}"

        # Load the model
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)

        # Text input area with analyze button (using columns for better layout)
        col1, col2 = st.columns([4, 1])

        with col1:
            user_text = st.text_area(
                "Enter text to analyze",
                "Joko Widodo adalah Presiden Indonesia yang saat ini sedang berada di Jakarta.",
                height=150,
            )

        # Add a prominent analysis button
        with col2:
            st.write("")  # Add space for alignment
            st.write("")  # Add space for alignment
            analyze_button = st.button(
                "✨ Analyze Text", use_container_width=True, type="primary"
            )

        if user_text and analyze_button:
            # Process text input
            with st.spinner("Processing text..."):
                # Load SpaCy model for tokenization and POS tagging
                try:
                    nlp = spacy.load("id_nusantara")
                except:
                    st.warning("Downloading id_nusantara model...")
                    try:
                        spacy.cli.download("id_nusantara")
                        nlp = spacy.load("id_nusantara")
                    except Exception as e:
                        st.error(f"Error loading SpaCy model: {e}")
                        nlp = spacy.blank("id")  # Fallback to blank model

                # Process with SpaCy
                doc = nlp(user_text)

                # Convert to the format needed for CRF
                tokens = [token.text for token in doc]
                pos_tags = [token.pos_ for token in doc]

                # Create a mock sentence for feature extraction
                mock_sentence = [
                    (token, pos, "O") for token, pos in zip(tokens, pos_tags)
                ]

                # Extract features and predict
                features = sent2features(mock_sentence)
                predictions = tagger.tag(features)

                # Display the analyzed text
                st.subheader("Analysis Results")

                # Display text with POS tags
                st.subheader("POS Tags")
                pos_html = ""
                for token, pos in zip(tokens, pos_tags):
                    pos_html += f"""
                    <div style='display:inline-block; margin-right:10px; margin-bottom:8px; text-align:center;'>
                        <div style='padding:5px 8px; background-color:#ffffff; border:1px solid #dddddd; border-radius:4px 4px 0 0; font-weight:bold;'>{token}</div>
                        <div style='padding:3px 8px; background-color:#4b5563; color:white; border-radius:0 0 4px 4px; font-size:0.8em; font-weight:bold;'>{pos}</div>
                    </div>
                    """
                st.html(f"<div style='line-height:2.5;'>{pos_html}</div>")

                # Count entity types
                entity_counts = {}
                for tag in predictions:
                    if tag != "O":
                        entity_type = tag.split("-")[1] if "-" in tag else tag
                        entity_counts[entity_type] = (
                            entity_counts.get(entity_type, 0) + 1
                        )

                # Show entity counts
                if entity_counts:
                    st.subheader("Entity Counts")
                    cols = st.columns(min(len(entity_counts), 4))
                    for i, (entity, count) in enumerate(entity_counts.items()):
                        with cols[i % len(cols)]:
                            st.metric(entity, count)

                # Visualize the predictions
                st.subheader("Named Entities")
                st.html(
                    visualize_ner_blocks(tokens, predictions, "Identified Entities")
                )
