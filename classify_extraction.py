import pandas as pd
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json


def classify_extraction_model(paper_row, client, model):
    """Classify the data extraction model/method used in the paper."""

    classification_prompt = """
    # Materials Data Extraction Model Classification

    ## Task
    Identify and classify the data/information extraction model(s) or method(s) used in this paper.

    ## Classification Categories

    ### 1. Large Language Models (LLMs)
    - GPT series (GPT-3, GPT-4, ChatGPT, GPT-3.5)
    - BERT variants (BERT, SciBERT, MatBERT, MaterialsBERT, BioBERT, ChemBERT)
    - T5 variants (T5, SciFive, MaterialsT5)
    - LLaMA series (LLaMA, LLaMA-2, Alpaca)
    - Other LLMs (PaLM, Claude, Gemini, Mistral, Qwen, Baichuan)

    ### 2. Traditional Deep Learning
    - LSTM/BiLSTM/GRU
    - CNN (Convolutional Neural Networks)
    - RNN (Recurrent Neural Networks)
    - Attention mechanisms (non-transformer based)

    ### 3. Transformer-based (Non-LLM)
    - Custom transformer architectures
    - Transformer for NER/Relation Extraction
    - Seq2Seq transformers

    ### 4. Classical Machine Learning
    - CRF (Conditional Random Fields)
    - SVM (Support Vector Machines)
    - Random Forest/Decision Trees
    - Naive Bayes
    - Maximum Entropy

    ### 5. Rule-Based & Hybrid
    - Regular expressions/Pattern matching
    - Dictionary/Lexicon-based
    - Rule-based parsers (ChemDataExtractor, etc.)
    - Hybrid (combining ML and rules)

    ### 6. Knowledge Graph & Ontology
    - Knowledge graph methods
    - Ontology-driven extraction
    - Graph Neural Networks (GNN) for extraction
    - Knowledge base integration

    ### 7. Multi-Modal
    - Vision-Language models
    - Text + Image/Structure extraction
    - Multi-modal transformers

    ### 8. Other/Unspecified
    - Novel/custom architectures
    - Method not clearly specified
    - Multiple methods without clear primary

    ## Output Format
    Return a JSON object with the following structure:
    {
        "primary_category": "category name",
        "specific_models": ["model1", "model2"],
        "secondary_tags": ["tag1", "tag2"],
        "confidence": "high/medium/low"
    }

    ### Secondary Tags (select all that apply):
    - "fine-tuned": Model was fine-tuned on domain data
    - "pre-trained": Uses pre-trained models
    - "domain-specific": Domain-specific model for materials
    - "end-to-end": End-to-end learning approach
    - "pipeline": Pipeline-based approach
    - "zero-shot": Zero-shot or few-shot learning
    - "ensemble": Uses ensemble methods

    ### Confidence Levels:
    - "high": Model/method clearly described in abstract/title
    - "medium": Model/method implied or partially described
    - "low": Model/method unclear or ambiguous

    ## Important Notes:
    - If multiple methods are used, select the PRIMARY method
    - List all SPECIFIC model names mentioned (e.g., "GPT-4", "SciBERT")
    - If no specific model is mentioned but approach is clear, use category
    - For rule-based or classical methods, still capture the approach

    ## Examples:

    Example 1:
    Title: "MatBERT: A materials science language model for information extraction"
    → {"primary_category": "Large Language Models (LLMs)", "specific_models": ["MatBERT", "BERT"], 
       "secondary_tags": ["pre-trained", "domain-specific", "fine-tuned"], "confidence": "high"}

    Example 2:
    Title: "Text mining of materials synthesis procedures using BiLSTM-CRF"
    → {"primary_category": "Traditional Deep Learning", "specific_models": ["BiLSTM", "CRF"], 
       "secondary_tags": ["pipeline"], "confidence": "high"}

    Example 3:
    Title: "ChemDataExtractor: A toolkit for automated extraction of chemical information"
    → {"primary_category": "Rule-Based & Hybrid", "specific_models": ["ChemDataExtractor"], 
       "secondary_tags": ["pipeline", "hybrid"], "confidence": "high"}

    Now analyze the paper and return ONLY the JSON object.
    """

    messages = [
        {"role": "system",
         "content": "You are an expert in NLP and materials informatics. Analyze papers to identify "
                    "the data extraction models/methods used. Return ONLY valid JSON."},
        {"role": "user", "content": f"""
        Analyze the following paper:

        Title: {paper_row.get('Article Title', 'N/A')}
        Keywords: {paper_row.get('Author Keywords', 'N/A')} | {paper_row.get('Keywords Plus', 'N/A')}
        Abstract: {paper_row.get('Abstract', 'N/A')}

        Classification Guide:
        {classification_prompt}

        Return ONLY the JSON object with classification results.
        """
         }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )

        tokens_used = getattr(completion.usage, "total_tokens", "Unknown")
        response_text = completion.choices[0].message.content.strip()

        # Try to parse JSON
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        result = json.loads(response_text)

        print(f"Tokens used: {tokens_used}")
        print(f"Classified: {paper_row.get('Article Title', 'N/A')[:60]}...")
        print(f"Category: {result.get('primary_category', 'Unknown')}")
        print(f"Models: {result.get('specific_models', [])}\n")

        return result

    except json.JSONDecodeError as e:
        print(f"JSON parsing error for paper {paper_row.name}: {e}")
        print(f"Response was: {response_text[:200]}")
        return {
            "primary_category": "Other/Unspecified",
            "specific_models": [],
            "secondary_tags": [],
            "confidence": "low"
        }
    except Exception as e:
        print(f"Error processing paper {paper_row.name}: {e}")
        return {
            "primary_category": "Other/Unspecified",
            "specific_models": [],
            "secondary_tags": [],
            "confidence": "low"
        }


def batch_classify_models(input_csv="classified_papers.csv",
                          output_csv="classified_with_models.csv",
                          max_workers=5,
                          only_relevant=True):
    """
    Batch classify extraction models used in papers.

    Args:
        input_csv: Input CSV with relevance classification
        output_csv: Output CSV with model classification
        max_workers: Number of parallel workers
        only_relevant: Only process papers marked as relevant (is_relevant=1)
    """
    # Initialize API client
    api_key = os.getenv('DEEPSEEK_API_KEY')
    base_url = "https://api.deepseek.com/"
    model = "deepseek-chat"

    client = OpenAI(
        api_key=f"{api_key}",
        base_url=base_url,
    )

    # Read input data
    try:
        df = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
        print(f"Loaded {len(df)} papers from {input_csv}")

        # Filter for relevant papers only if specified
        if only_relevant and 'is_relevant' in df.columns:
            df_to_process = df[df['is_relevant'] == 1].copy()
            print(f"Filtering to {len(df_to_process)} relevant papers")
        else:
            df_to_process = df.copy()

    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    # Initialize new columns
    df['model_primary_category'] = None
    df['model_specific_models'] = None
    df['model_secondary_tags'] = None
    df['model_confidence'] = None

    print_lock = Lock()

    def process_single_paper(idx, row):
        """Process a single paper and return classification."""
        try:
            result = classify_extraction_model(row, client, model)
            with print_lock:
                print(f"✓ Classified paper {idx + 1}")
            return idx, result
        except Exception as e:
            with print_lock:
                print(f"✗ Error classifying paper {idx + 1}: {e}")
            return idx, {
                "primary_category": "Other/Unspecified",
                "specific_models": [],
                "secondary_tags": [],
                "confidence": "low"
            }

    print(f"\nStarting model classification with {max_workers} workers...\n")

    # Get indices to process
    indices_to_process = df_to_process.index.tolist()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_paper, idx, df.loc[idx]): idx
            for idx in indices_to_process
        }

        completed = 0
        for future in as_completed(futures):
            idx, result = future.result()

            # Update dataframe with results
            df.at[idx, 'model_primary_category'] = result.get('primary_category', 'Other/Unspecified')
            df.at[idx, 'model_specific_models'] = json.dumps(result.get('specific_models', []))
            df.at[idx, 'model_secondary_tags'] = json.dumps(result.get('secondary_tags', []))
            df.at[idx, 'model_confidence'] = result.get('confidence', 'low')

            completed += 1

            if completed % 10 == 0:
                with print_lock:
                    print(
                        f"\nProgress: {completed}/{len(indices_to_process)} papers classified ({completed / len(indices_to_process) * 100:.1f}%)\n")

    # Save output
    try:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n{'=' * 60}")
        print(f"✓ Successfully saved results to {output_csv}")
        print(f"{'=' * 60}")

        # Print statistics
        print(f"\nModel Classification Statistics:")
        print("\nPrimary Categories:")
        print(df[df['model_primary_category'].notna()]['model_primary_category'].value_counts())

        print(f"\nConfidence Distribution:")
        print(df[df['model_confidence'].notna()]['model_confidence'].value_counts())

    except Exception as e:
        print(f"✗ Failed to save results: {e}")


def analyze_model_distribution(csv_file="classified_with_models.csv"):
    """
    Analyze and visualize the distribution of extraction models.
    """
    df = pd.read_csv(csv_file, low_memory=False, encoding="utf-8-sig")

    # Filter classified papers
    classified = df[df['model_primary_category'].notna()]

    print(f"\n{'=' * 60}")
    print(f"MODEL CLASSIFICATION ANALYSIS")
    print(f"{'=' * 60}\n")

    print(f"Total papers analyzed: {len(df)}")
    print(f"Papers with model classification: {len(classified)}")
    print(f"Classification rate: {len(classified) / len(df) * 100:.1f}%\n")

    print(f"{'=' * 60}")
    print(f"PRIMARY CATEGORY DISTRIBUTION")
    print(f"{'=' * 60}")
    category_counts = classified['model_primary_category'].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(classified) * 100
        print(f"{category:.<50} {count:>4} ({percentage:>5.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"CONFIDENCE DISTRIBUTION")
    print(f"{'=' * 60}")
    conf_counts = classified['model_confidence'].value_counts()
    for conf, count in conf_counts.items():
        percentage = count / len(classified) * 100
        print(f"{conf:.<50} {count:>4} ({percentage:>5.1f}%)")

    # Analyze specific models
    print(f"\n{'=' * 60}")
    print(f"TOP SPECIFIC MODELS (mentioned at least 3 times)")
    print(f"{'=' * 60}")

    all_models = []
    for models_str in classified['model_specific_models'].dropna():
        try:
            models = json.loads(models_str)
            all_models.extend(models)
        except:
            pass

    if all_models:
        model_counts = pd.Series(all_models).value_counts()
        for model, count in model_counts.head(20).items():
            if count >= 3:
                print(f"{model:.<50} {count:>4}")

    # Analyze secondary tags
    print(f"\n{'=' * 60}")
    print(f"SECONDARY TAG DISTRIBUTION")
    print(f"{'=' * 60}")

    all_tags = []
    for tags_str in classified['model_secondary_tags'].dropna():
        try:
            tags = json.loads(tags_str)
            all_tags.extend(tags)
        except:
            pass

    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts()
        for tag, count in tag_counts.items():
            percentage = count / len(classified) * 100
            print(f"{tag:.<50} {count:>4} ({percentage:>5.1f}%)")


if __name__ == "__main__":
    # Step 1: Classify models in relevant papers
    batch_classify_models(
        input_csv=r"G:\2026-01-26 材料信息提取\combined_classified.csv",
        output_csv=r"G:\2026-01-26 材料信息提取\combined_classified_with_models.csv",
        max_workers=16,
        only_relevant=True  # Only process relevant papers
    )

    # Step 2: Analyze the distribution
    analyze_model_distribution(r"G:\2026-01-26 材料信息提取\combined_classified_with_models.csv")