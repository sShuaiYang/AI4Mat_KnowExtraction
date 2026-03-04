# -*- coding: utf-8 -*-
"""
@Time ： 2026/1/28 14:01
@Auth ： Shuai
@File ：classify_materials.py
@IDE ：PyCharm
"""
import pandas as pd
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json


def classify_materials(paper_row, client, model):
    """Classify the material types discussed in the paper."""

    classification_prompt = """
    # Materials Classification Task

    ## Objective
    Identify ALL material types that are the PRIMARY focus of this paper's research.

    ## Material Categories (Select ALL that apply)

    1. **Nanoparticles**
       - Metal nanoparticles (Au, Ag, Pt, Cu nanoparticles)
       - Metal oxide nanoparticles (TiO2, ZnO, Fe3O4 nanoparticles)
       - Quantum dots, nanospheres, nanoclusters
       - Core-shell nanoparticles
       - **EXCLUDE**: Nanowires, nanotubes, 2D nanosheets (these belong to "2D Materials")
       - **Key indicator**: Specifically mentions "nanoparticles", "NPs", "quantum dots", "QDs", "nanospheres"

    2. **Battery Cathode Materials**
       - Lithium-ion cathode: LiCoO2 (LCO), LiNixCoyMnzO2 (NCM), LiNixCoyAlzO2 (NCA), 
         LiFePO4 (LFP), LiMn2O4 (LMO), Li-rich materials
       - Sodium-ion cathode materials
       - Other battery cathode/positive electrode materials
       - **Key indicator**: "cathode", "positive electrode", battery material names (NCM, LFP, LCO)

    3. **Photocatalytic Materials**
       - TiO2 (titanium dioxide), ZnO, g-C3N4 (graphitic carbon nitride)
       - BiVO4, CdS, perovskites for photocatalysis
       - Visible-light photocatalysts, UV photocatalysts
       - Materials for water splitting, CO2 reduction, pollutant degradation via photocatalysis
       - **Key indicator**: "photocatalyst", "photocatalytic", "water splitting", "photodegradation"

    4. **Polymers**
       - Organic polymers: polyethylene, polypropylene, PVC, PMMA, polystyrene
       - Conductive polymers: PEDOT:PSS, polyaniline, polypyrrole
       - Polymer electrolytes for batteries
       - Biopolymers: chitosan, cellulose, collagen, PLGA
       - Polymer composites, copolymers
       - **Key indicator**: "polymer", "polyethylene", "PMMA", "resin", polymer names

    5. **Thermoelectric Materials**
       - Bi2Te3, Sb2Te3, PbTe, SnSe
       - Skutterudites, half-Heusler alloys, Zintl phases
       - Organic thermoelectric materials
       - Materials with Seebeck coefficient, ZT value mentioned
       - **Key indicator**: "thermoelectric", "Seebeck", "ZT", "Bi2Te3", "PbTe"

    6. **MOFs (Metal-Organic Frameworks)**
       - Metal-Organic Frameworks: ZIF-8, UiO-66, HKUST-1, MIL-101
       - Covalent Organic Frameworks (COFs)
       - Porous coordination polymers
       - **Key indicator**: "MOF", "metal-organic framework", "ZIF", "UiO", "COF"

    7. **Biomaterials**
       - Tissue engineering scaffolds, bone substitutes
       - Biocompatible materials, bioceramics (hydroxyapatite)
       - Drug delivery systems, biodegradable materials
       - Medical implant materials
       - **Key indicator**: "biomaterial", "biocompatible", "tissue engineering", "drug delivery", 
         "hydroxyapatite", "scaffold"

    8. **2D Materials**
       - Graphene, graphene oxide, reduced graphene oxide
       - Transition metal dichalcogenides (TMDs): MoS2, WS2, WSe2, MoSe2
       - MXenes: Ti3C2Tx, Ti2CTx
       - Hexagonal boron nitride (h-BN), phosphorene (black phosphorus)
       - Other layered 2D materials
       - **Key indicator**: "graphene", "MoS2", "MXene", "2D material", "monolayer", "few-layer"

    9. **Semiconductor Materials**
       - Elemental: Silicon, Germanium
       - III-V compounds: GaN, GaAs, InP, AlGaN
       - II-VI compounds: CdTe, CdSe, ZnS, ZnSe
       - Organic semiconductors
       - Perovskite semiconductors (for electronics, not just photocatalysis)
       - **Key indicator**: "semiconductor", "Si wafer", "GaN", "GaAs", "p-type", "n-type", "bandgap"

    10. **Other Materials**
        - Metals and alloys (if not nanoparticles)
        - Ceramics (if not fitting other categories)
        - Magnetic materials
        - Optical materials
        - Novel materials not fitting above categories
        - **Use this only if**: Material doesn't clearly fit any specific category above

    ## Classification Rules

    1. **Select ALL applicable categories** - A paper can belong to multiple categories
    2. **Primary focus only** - Only select if the material is a PRIMARY research subject
    3. **Be specific**: 
       - If graphene is studied → select "2D Materials", NOT "Nanoparticles"
       - If TiO2 for photocatalysis → select "Photocatalytic Materials"
       - If TiO2 nanoparticles for photocatalysis → select BOTH "Nanoparticles" AND "Photocatalytic Materials"
       - If LiFePO4 cathode → select "Battery Cathode Materials"
    4. **Distinction between categories**:
       - Nanoparticles = 0D nanostructures (spherical, dots)
       - 2D Materials = 1-few layer sheet materials
       - Don't confuse "nanostructured" with "nanoparticles"
    5. **Confidence scoring**:
       - "high": Materials clearly and explicitly described in title/abstract
       - "medium": Materials implied or indirectly mentioned
       - "low": Materials unclear or ambiguous

    ## Output Format

    Return a JSON object:
    {
        "material_categories": ["category1", "category2", ...],
        "confidence": "high/medium/low",
        "primary_category": "most dominant single category"
    }

    ## Examples

    Example 1:
    Title: "Text mining of LiFePO4 synthesis procedures for lithium-ion batteries"
    → {
        "material_categories": ["Battery Cathode Materials"],
        "confidence": "high",
        "primary_category": "Battery Cathode Materials"
    }

    Example 2:
    Title: "Automated extraction of graphene synthesis parameters using NLP"
    → {
        "material_categories": ["2D Materials"],
        "confidence": "high",
        "primary_category": "2D Materials"
    }

    Example 3:
    Title: "Machine learning for TiO2 nanoparticle photocatalytic properties"
    → {
        "material_categories": ["Photocatalytic Materials", "Nanoparticles"],
        "confidence": "high",
        "primary_category": "Photocatalytic Materials"
    }

    Example 4:
    Title: "Data extraction for MOF synthesis and gas adsorption properties"
    → {
        "material_categories": ["MOFs"],
        "confidence": "high",
        "primary_category": "MOFs"
    }

    Example 5:
    Title: "NLP-based mining of MoS2 synthesis for electronic applications"
    → {
        "material_categories": ["2D Materials", "Semiconductor Materials"],
        "confidence": "high",
        "primary_category": "2D Materials"
    }

    Example 6:
    Title: "Automated extraction of thermoelectric properties from Bi2Te3 literature"
    → {
        "material_categories": ["Thermoelectric Materials"],
        "confidence": "high",
        "primary_category": "Thermoelectric Materials"
    }

    Example 7:
    Title: "Text mining for polymer electrolyte materials in solid-state batteries"
    → {
        "material_categories": ["Polymers"],
        "confidence": "high",
        "primary_category": "Polymers"
    }

    Now analyze the paper and return ONLY the JSON object.
    """

    messages = [
        {"role": "system",
         "content": "You are an expert in materials science classification. Analyze papers to identify "
                    "material types based on the given classification system. Return ONLY valid JSON."},
        {"role": "user", "content": f"""
        Analyze the following paper:

        Title: {paper_row.get('Article Title', 'N/A')}
        Keywords: {paper_row.get('Author Keywords', 'N/A')} | {paper_row.get('Keywords Plus', 'N/A')}
        Abstract: {paper_row.get('Abstract', 'N/A')}

        Classification Guide:
        {classification_prompt}

        Return ONLY the JSON object with material classifications.
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

        # Clean response
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        print(f"Tokens used: {tokens_used}")
        print(f"Classified: {paper_row.get('Article Title', 'N/A')[:60]}...")
        print(f"Categories: {result.get('material_categories', [])}")
        print(f"Primary: {result.get('primary_category', 'Unknown')}\n")

        return result

    except json.JSONDecodeError as e:
        print(f"JSON parsing error for paper {paper_row.name}: {e}")
        print(f"Response was: {response_text[:200]}")
        return {
            "material_categories": ["Other Materials"],
            "confidence": "low",
            "primary_category": "Other Materials"
        }
    except Exception as e:
        print(f"Error processing paper {paper_row.name}: {e}")
        return {
            "material_categories": ["Other Materials"],
            "confidence": "low",
            "primary_category": "Other Materials"
        }


def batch_classify_materials(input_csv="classified_with_models.csv",
                             output_csv="classified_with_materials.csv",
                             max_workers=5,
                             only_relevant=True):
    """
    Batch classify material types in papers.

    Args:
        input_csv: Input CSV file
        output_csv: Output CSV file
        max_workers: Number of parallel workers
        only_relevant: Only process papers marked as relevant
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

        # Filter for relevant papers if specified
        if only_relevant and 'is_relevant' in df.columns:
            df_to_process = df[df['is_relevant'] == 1].copy()
            print(f"Filtering to {len(df_to_process)} relevant papers")
        else:
            df_to_process = df.copy()

    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    # Initialize new columns
    df['material_categories'] = None
    df['material_confidence'] = None
    df['primary_material_category'] = None

    print_lock = Lock()

    def process_single_paper(idx, row):
        """Process a single paper and return classification."""
        try:
            result = classify_materials(row, client, model)
            with print_lock:
                print(f"✓ Classified paper {idx + 1}")
            return idx, result
        except Exception as e:
            with print_lock:
                print(f"✗ Error classifying paper {idx + 1}: {e}")
            return idx, {
                "material_categories": ["Other Materials"],
                "confidence": "low",
                "primary_category": "Other Materials"
            }

    print(f"\nStarting material classification with {max_workers} workers...\n")

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
            df.at[idx, 'material_categories'] = json.dumps(result.get('material_categories', []))
            df.at[idx, 'material_confidence'] = result.get('confidence', 'low')
            df.at[idx, 'primary_material_category'] = result.get('primary_category', 'Other Materials')

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

    except Exception as e:
        print(f"✗ Failed to save results: {e}")


def analyze_material_distribution(csv_file="classified_with_materials.csv",
                                  output_stats="material_statistics.txt"):
    """
    Analyze and report material classification statistics.
    """
    df = pd.read_csv(csv_file, low_memory=False, encoding="utf-8-sig")

    # Filter classified papers
    classified = df[df['primary_material_category'].notna()]

    report = []
    report.append("=" * 80)
    report.append("MATERIAL CLASSIFICATION ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    report.append(f"Total papers in dataset: {len(df)}")
    report.append(f"Papers with material classification: {len(classified)}")
    report.append(f"Classification rate: {len(classified) / len(df) * 100:.1f}%")
    report.append("")

    # Primary category distribution
    report.append("=" * 80)
    report.append("PRIMARY MATERIAL CATEGORY DISTRIBUTION")
    report.append("=" * 80)

    primary_counts = classified['primary_material_category'].value_counts()
    for category, count in primary_counts.items():
        percentage = count / len(classified) * 100
        bar = "█" * int(percentage / 2)
        report.append(f"{category:<35} {count:>4} ({percentage:>5.1f}%) {bar}")

    report.append("")

    # All categories distribution (since papers can have multiple categories)
    report.append("=" * 80)
    report.append("ALL MATERIAL CATEGORIES (papers can have multiple)")
    report.append("=" * 80)

    all_categories = []
    for cats_str in classified['material_categories'].dropna():
        try:
            cats = json.loads(cats_str)
            all_categories.extend(cats)
        except:
            pass

    if all_categories:
        category_counts = pd.Series(all_categories).value_counts()
        total_mentions = len(all_categories)
        for category, count in category_counts.items():
            percentage = count / len(classified) * 100
            mention_pct = count / total_mentions * 100
            bar = "█" * int(percentage / 2)
            report.append(
                f"{category:<35} {count:>4} ({percentage:>5.1f}% of papers, {mention_pct:>5.1f}% of mentions) {bar}")

    report.append("")

    # Confidence distribution
    report.append("=" * 80)
    report.append("CONFIDENCE DISTRIBUTION")
    report.append("=" * 80)

    conf_counts = classified['material_confidence'].value_counts()
    for conf, count in conf_counts.items():
        percentage = count / len(classified) * 100
        report.append(f"{conf:<35} {count:>4} ({percentage:>5.1f}%)")

    report.append("")

    # Multi-category analysis
    report.append("=" * 80)
    report.append("MULTI-CATEGORY ANALYSIS")
    report.append("=" * 80)

    category_lengths = []
    for cats_str in classified['material_categories'].dropna():
        try:
            cats = json.loads(cats_str)
            category_lengths.append(len(cats))
        except:
            pass

    if category_lengths:
        length_counts = pd.Series(category_lengths).value_counts().sort_index()
        report.append("Number of material categories per paper:")
        for length, count in length_counts.items():
            percentage = count / len(category_lengths) * 100
            report.append(
                f"  {length} {'category' if length == 1 else 'categories'}: {count:>4} papers ({percentage:>5.1f}%)")

    report.append("")

    # Category co-occurrence analysis
    report.append("=" * 80)
    report.append("COMMON CATEGORY COMBINATIONS (appearing 3+ times)")
    report.append("=" * 80)

    combinations = []
    for cats_str in classified['material_categories'].dropna():
        try:
            cats = json.loads(cats_str)
            if len(cats) > 1:
                combinations.append(tuple(sorted(cats)))
        except:
            pass

    if combinations:
        combo_counts = pd.Series(combinations).value_counts()
        for combo, count in combo_counts.items():
            if count >= 3:
                combo_str = " + ".join(combo)
                report.append(f"{combo_str:<60} {count:>4}")

    report.append("")
    report.append("=" * 80)

    # Print to console
    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    try:
        with open(output_stats, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ Statistics saved to {output_stats}")
    except Exception as e:
        print(f"✗ Failed to save statistics: {e}")

    return df


def export_by_material_category(csv_file="classified_with_materials.csv",
                                output_dir="material_categories"):
    """
    Export separate CSV files for each material category.
    """
    import os

    df = pd.read_csv(csv_file, low_memory=False, encoding="utf-8-sig")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique categories
    all_categories = set()
    for cats_str in df['material_categories'].dropna():
        try:
            cats = json.loads(cats_str)
            all_categories.update(cats)
        except:
            pass

    print(f"\nExporting papers by material category to '{output_dir}/'...")
    print("=" * 80)

    # Export for each category
    category_stats = []
    for category in sorted(all_categories):
        # Find papers with this category
        mask = df['material_categories'].apply(
            lambda x: category in json.loads(x) if pd.notna(x) and x != 'null' else False
        )

        category_df = df[mask]

        if len(category_df) > 0:
            # Clean filename
            filename = category.replace("/", "-").replace(" ", "_") + ".csv"
            filepath = os.path.join(output_dir, filename)

            category_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            category_stats.append((category, len(category_df), filename))
            print(f"  ✓ {category:<40} {len(category_df):>4} papers → {filename}")

    print("=" * 80)
    print(f"✓ Exported {len(all_categories)} category files")
    print(f"✓ Total papers exported: {sum(stat[1] for stat in category_stats)} (with duplicates across categories)")

    return category_stats


if __name__ == "__main__":
    # Step 1: Classify materials
    print("Step 1: Classifying materials...")
    print("=" * 80)
    batch_classify_materials(
        input_csv=r"G:\2026-01-26 材料信息提取\combined_classified_with_models.csv",
        output_csv=r"G:\2026-01-26 材料信息提取\combined_classified_with_materials.csv",
        max_workers=24,
        only_relevant=True
    )

    # Step 2: Analyze distribution
    print("\n" + "=" * 80)
    print("Step 2: Analyzing material distribution...")
    print("=" * 80 + "\n")
    analyze_material_distribution(
        csv_file=r"G:\2026-01-26 材料信息提取\combined_classified_with_materials.csv",
        output_stats="material_statistics.txt"
    )

    # # Step 3: Export by category
    # print("\n" + "=" * 80)
    # print("Step 3: Exporting papers by category...")
    # print("=" * 80 + "\n")
    # export_by_material_category(
    #     csv_file="classified_with_materials.csv",
    #     output_dir="material_categories"
    # )