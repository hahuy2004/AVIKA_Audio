"""
Query Generator using LLM (GPT-4) for Offline Pre-processing
Phase 1: Generate enriched query variations
"""

import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

# PROMPT_TEMPLATE = """You are given a caption describing a visual scene. Your task is to rewrite the caption into {n} different sentences following the rules:
# 1. You can diversify the sentence structure and word usage, but you should strictly keep the same semantic meaning.
# 2. Do not add uncertain details that do not associate with the visual scene. The rewriting should strictly follow the factual information in the original caption.
# 3. The rewritten captions should be diverse in number of words.
# 4. The rewritten captions should be no more than 10 words longer than the original caption.

# The input caption is: {caption}

# Please output ONLY the {n} rewritten sentences, one per line, without numbering or any additional text."""

# ==================

# MSRVTT
# PROMPT_TEMPLATE = """You are given a caption describing a visual scene.

# Your task is to generate EXACTLY {n} rewritten versions of the caption, following the constraints below:

# 1. Each rewritten sentence must preserve the exact semantic meaning of the original caption.
# 2. Do NOT introduce any new objects, actions, attributes, intentions, or contextual details that are not explicitly stated in the original caption.
# 3. The rewriting must be strictly grounded in the visual content described by the original caption.
# 4. Sentence structure and lexical choices should be diversified across rewritten sentences.
# 5. The rewritten captions should vary in length and number of words.
# 6. Each rewritten caption must be no more than 10 words longer than the original caption.
# 7. The original caption must NOT appear in the rewritten outputs.
# 8. No two rewritten captions may be identical.
# 9. Do NOT use commas (,), quotation marks ("), semicolons (;), or colons (:).
# 10. Each rewritten caption must be a single simple sentence without punctuation inside the sentence.
# 11. Use conjunctions such as "and", "while", or "as" instead of punctuation.

# The input caption is:
# "{caption}"

# Output ONLY the {n} rewritten sentences, one per line, without numbering, bullet points, or any additional explanations.
# """

# DiDeMo
PROMPT_TEMPLATE = """You are given a caption describing a visual scene, which often consists of multiple sentences or a sequence of events.

Your task is to generate EXACTLY {n} rewritten versions of the caption, following the constraints below:

1. Each rewritten version must preserve the exact semantic meaning and the complete sequence of events from the original caption.
2. Do NOT introduce any new objects, actions, attributes, intentions, or contextual details that are not explicitly stated.
3. The rewriting must be strictly grounded in the visual content described by the original caption.
4. Sentence structure and lexical choices should be diversified across rewritten versions.
5. Maintain a similar length and level of detail as the original caption. Do NOT over-summarize or drop any events.
6. The original caption must NOT appear in the rewritten outputs.
7. No two rewritten versions may be identical.
8. You may use periods (.) and commas (,) to clearly separate multiple actions, but each complete rewritten version MUST be output on a single continuous line.
9. Do NOT use quotation marks ("), semicolons (;), or colons (:).

The input caption is:
"{caption}"

Output ONLY the {n} rewritten versions, one per line, without numbering, bullet points, or any additional explanations.
"""

def generate_enriched_queries(
    input_captions,
    output_json_path,
    api_key,
    n_variations=10,
    model="gpt-4.1",
    batch_size=1,
    sleep_time=1.0
):
    """
    Generate enriched query variations using GPT-4.1.
    
    Args:
        input_captions: List of original captions or dict with video_id -> caption
        output_json_path: Path to save enriched data
        api_key: OpenAI API key
        n_variations: Number of variations to generate (default: 10)
        model: OpenAI model to use (default: gpt-4.1)
        batch_size: Process captions in batches (default: 1)
        sleep_time: Sleep between API calls to avoid rate limits
        
    Returns:
        dict: {video_id: [original_caption, variation1, ..., variation_n]}
    """
    client = OpenAI(api_key=api_key)
    
    # Convert to dict format if input is list
    if isinstance(input_captions, list):
        input_captions = {f"video_{i}": cap for i, cap in enumerate(input_captions)}
    
    enriched_data = {}
    
    # Check if output file exists for resuming
    if os.path.exists(output_json_path):
        print(f"Loading existing enriched data from {output_json_path}")
        with open(output_json_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
    
    total_captions = len(input_captions)
    already_processed = len(enriched_data)
    remaining = total_captions - already_processed
    
    print(f"Generating enriched queries for {total_captions} captions...")
    print(f"Using model: {model}, variations per caption: {n_variations}")
    if already_processed > 0:
        print(f"Already processed: {already_processed}, Remaining: {remaining}")
    
    processed_count = 0
    for video_id, original_caption in tqdm(input_captions.items(), desc="Enriching captions"):
        # Skip if already processed
        if video_id in enriched_data:
            continue
        
        processed_count += 1
        
        try:
            prompt = PROMPT_TEMPLATE.format(n=n_variations, caption=original_caption)
            
            # MSRVTT: 400
            # DiDeMo: 750
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=750
            )
            
            # Parse response
            variations_text = response.output_text.strip()
            
            # Parse and clean variations
            # Split by newline and strip each line
            raw_variations = [
                line.strip() 
                for line in variations_text.split('\n') 
                if line.strip()  # Remove empty lines
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            variations = []
            for var in raw_variations:
                # Normalize for comparison (lowercase, remove extra spaces)
                normalized = ' '.join(var.lower().split())
                original_normalized = ' '.join(original_caption.lower().split())
                
                # Skip if duplicate or matches original caption
                if normalized not in seen and normalized != original_normalized:
                    variations.append(var)
                    seen.add(normalized)
            
            # Ensure we have exactly n variations
            if len(variations) < n_variations:
                shortage = n_variations - len(variations)
                print(f"Warning: Only got {len(variations)} unique variations for {video_id}, padding {shortage} with modified original")
                # Pad with slightly modified original to reach n_variations
                for i in range(shortage):
                    variations.append(original_caption)
            elif len(variations) > n_variations:
                variations = variations[:n_variations]
            
            # Store as [original, var1, var2, ..., var_n]
            enriched_data[video_id] = [original_caption] + variations
            
            # Save periodically
            if processed_count % 10 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Progress: {len(enriched_data)}/{total_captions} completed, saved checkpoint")
            
            # Rate limiting
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            # Store original only if error
            enriched_data[video_id] = [original_caption] * (n_variations + 1)
            continue
    
    # Final save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    # Statistics
    total_captions = sum(len(caps) for caps in enriched_data.values())
    avg_per_video = total_captions / len(enriched_data) if enriched_data else 0
    
    print(f"\n✅ Enrichment completed!")
    print(f"Enriched data saved to {output_json_path}")
    print(f"Total videos: {len(enriched_data)}")
    print(f"Total captions (including originals): {total_captions}")
    print(f"Average captions per video: {avg_per_video:.1f}")
    print(f"Expected: {n_variations + 1} (1 original + {n_variations} enriched)")
    
    return enriched_data


def load_enriched_queries(json_path):
    """
    Load pre-generated enriched queries from JSON file.
    
    Args:
        json_path: Path to enriched queries JSON
        
    Returns:
        dict: {video_id: [original_caption, variation1, ..., variation_n]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded enriched queries for {len(data)} videos from {json_path}")
    return data


if __name__ == "__main__":
    # Example usage for offline preprocessing
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enriched queries using GPT-4")
    parser.add_argument("--input_json", type=str, required=True, help="Input captions JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output enriched queries JSON file")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--n_variations", type=int, default=10, help="Number of variations per caption")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    # Load input captions
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_captions = json.load(f)
    
    # Generate enriched queries
    generate_enriched_queries(
        input_captions=input_captions,
        output_json_path=args.output_json,
        api_key=args.api_key,
        n_variations=args.n_variations,
        model=args.model
    )
