# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/28 18:58
@Auth ： Shuai
@File ：is_material_data_extraction_relevant.py
@IDE ：PyCharm
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI
import os
from threading import Lock


def is_materials_data_extraction_relevant(paper_row, client, model):
    """Determine if a paper is relevant to materials data/information extraction."""

    optimized_prompt = """
    # Materials Data/Information Extraction Relevance Assessment

    ## Task
    Determine if the paper is relevant to **extracting accurate materials data or structured information extraction** based on its title, keywords, and abstract.

    ## Relevant (Output 1) if the paper meets ANY of the following:
    - Develops or applies methods for extracting materials properties, characteristics, or data from scientific literature.
    - Uses natural language processing (NLP), machine learning, or AI to extract structured materials information from text.
    - Focuses on text mining or data mining for materials science databases or knowledge graphs.
    - Develops automated systems for parsing materials synthesis procedures, experimental conditions, or characterization results.
    - Extracts materials composition, structure, properties, or performance metrics from papers or patents.
    - Builds or contributes to materials informatics databases through automated data extraction.
    - Develops named entity recognition (NER), relation extraction, or information extraction specifically for materials domain.
    - Uses large language models (LLMs) or deep learning for materials literature analysis and data extraction.
    - Focuses on standardizing or structuring unstructured materials data from publications.
    - Develops tools or platforms for automated materials data curation from literature.

    ## Not Relevant (Output 0) if:
    - Focuses only on materials characterization techniques without data extraction components.
    - General machine learning applications in materials science without information extraction focus.
    - Materials discovery or design without literature data extraction methods.
    - Experimental materials synthesis or testing without automated data extraction.
    - Pure theoretical materials modeling without text/data mining components.
    - Database development without automated extraction from literature (manual curation only).
    - General NLP or information extraction methods not applied to materials domain.

    ## Output
    Strictly output **only one digit**:
    - "1" if relevant.
    - "0" if not relevant.

    Do not explain or justify your answer.
    """

    messages = [
        {"role": "system",
         "content": "You are an expert in materials informatics and data extraction. You must analyze papers for relevance "
                    "to materials data/information extraction. Reply with ONLY 0 or 1."},
        {"role": "user", "content": f"""
        Evaluate the following paper for relevance.

        Title: {paper_row.get('Article Title', 'N/A')}
        Keywords: {paper_row.get('Author Keywords', 'N/A')} | {paper_row.get('Keywords Plus', 'N/A')}
        Abstract: {paper_row.get('Abstract', 'N/A')}

        Assessment Guide:
        {optimized_prompt}

        Provide ONLY one digit: 1 (relevant) or 0 (not relevant).
        """
         }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        # Check token usage (handling cases where it might be missing)
        tokens_used = getattr(completion.usage, "total_tokens", "Unknown")
        print(f"Total tokens used: {tokens_used}")
        decision = int(completion.choices[0].message.content.strip())
        print(f"Decision: {decision}\n"
              f"paper_row: {paper_row.get('Article Title', 'N/A')}")

        return 1 if decision == 1 else 0  # Force binary output
    except Exception as e:
        print(f"Error processing paper {paper_row.name}: {e}")
        return 0


def batch_process(input_csv="combined.csv", output_csv="classified_papers.csv", max_workers=5):
    """
    Batch process papers with parallel execution.

    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        max_workers: Maximum number of parallel workers (default: 5)
    """
    # Initialize API client
    # 配置参数
    api_key = os.getenv('DEEPSEEK_API_KEY')  # 替换为你的实际 API 密钥
    base_url = "https://api.deepseek.com/"  # 默认值或自定义值
    model = "deepseek-chat"  # 默认值或自定义值



    client = OpenAI(
        api_key=f"{api_key}",
        base_url=base_url,
    )

    # Read input data
    try:
        df = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
        print(f"Loaded {len(df)} papers from {input_csv}")
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    # 用于线程安全的打印
    print_lock = Lock()

    def process_single_paper(idx, row):
        """Process a single paper and return index and result."""
        try:
            result = is_materials_data_extraction_relevant(row, client, model)
            with print_lock:
                print(f"✓ Processed paper {idx + 1}/{len(df)}: {row.get('Article Title', 'N/A')[:50]}...")
            return idx, result
        except Exception as e:
            with print_lock:
                print(f"✗ Error processing paper {idx + 1}: {e}")
            return idx, 0  # Default to 0 if error occurs

    # Process papers in parallel
    results = [None] * len(df)  # Pre-allocate results list

    print(f"\nStarting parallel processing with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_paper, idx, row): idx
            for idx, row in df.iterrows()
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1

            # Progress update every 10 completions
            if completed % 10 == 0:
                with print_lock:
                    print(f"\nProgress: {completed}/{len(df)} papers completed ({completed / len(df) * 100:.1f}%)\n")

    # Add results to dataframe
    df['is_relevant'] = results

    # Save output
    try:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n{'=' * 60}")
        print(f"✓ Successfully saved results to {output_csv}")
        print(f"{'=' * 60}")
        print(f"\nRelevance Statistics:")
        print(df['is_relevant'].value_counts())
        print(
            f"\nRelevant papers: {df['is_relevant'].sum()} / {len(df)} ({df['is_relevant'].sum() / len(df) * 100:.1f}%)")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")


# 可选：添加断点续传功能
def batch_process_with_checkpoint(input_csv="combined.csv",
                                  output_csv="classified_papers.csv",
                                  checkpoint_csv="checkpoint.csv",
                                  max_workers=5):
    """
    Batch process with checkpoint support for resuming interrupted runs.
    """
    # Initialize API client (same as above)
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
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    # Check for existing checkpoint
    processed_indices = set()
    if os.path.exists(checkpoint_csv):
        try:
            checkpoint_df = pd.read_csv(checkpoint_csv, encoding="utf-8-sig")
            if 'is_relevant' in checkpoint_df.columns:
                df['is_relevant'] = checkpoint_df['is_relevant']
                processed_indices = set(checkpoint_df[checkpoint_df['is_relevant'].notna()].index)
                print(f"Resumed from checkpoint: {len(processed_indices)} papers already processed")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    # Initialize is_relevant column if not exists
    if 'is_relevant' not in df.columns:
        df['is_relevant'] = None

    print_lock = Lock()

    def process_single_paper(idx, row):
        """Process a single paper and return index and result."""
        try:
            result = is_materials_data_extraction_relevant(row, client, model)
            with print_lock:
                print(f"✓ Processed paper {idx + 1}/{len(df)}")
            return idx, result
        except Exception as e:
            with print_lock:
                print(f"✗ Error processing paper {idx + 1}: {e}")
            return idx, 0

    # Get papers that need processing
    papers_to_process = [(idx, row) for idx, row in df.iterrows() if idx not in processed_indices]

    if not papers_to_process:
        print("All papers already processed!")
        return

    print(f"\nProcessing {len(papers_to_process)} remaining papers with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_paper, idx, row): idx
            for idx, row in papers_to_process
        }

        completed = 0
        for future in as_completed(futures):
            idx, result = future.result()
            df.at[idx, 'is_relevant'] = result
            completed += 1

            # Save checkpoint every 50 papers
            if completed % 50 == 0:
                df.to_csv(checkpoint_csv, index=False, encoding="utf-8-sig")
                with print_lock:
                    print(f"\n💾 Checkpoint saved: {completed}/{len(papers_to_process)} papers completed\n")

    # Final save
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ Successfully saved final results to {output_csv}")
    print(f"\nRelevance Statistics:")
    print(df['is_relevant'].value_counts())

    # Clean up checkpoint file
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)
        print(f"✓ Checkpoint file removed")


if __name__ == "__main__":
    # 基础版本（无断点续传）
    batch_process(
        input_csv=r"G:\2026-01-26 材料信息提取\combined.csv",
        output_csv=r"G:\2026-01-26 材料信息提取\combined_classified.csv",
        max_workers=36  # 根据API限制调整
    )

    # 或使用带断点续传的版本
    # batch_process_with_checkpoint(
    #     input_csv="combined.csv",
    #     output_csv="classified_papers.csv",
    #     checkpoint_csv="checkpoint.csv",
    #     max_workers=10
    # )
