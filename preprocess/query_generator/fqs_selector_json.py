"""
Query Selection Algorithms for JSON format
Phase 2: Select k most diverse (FQS), similar (NQS), or random queries from augmented set

Usage FQS:
    python fqs_selector.py --k 2 --input_path data.json

Usage NQS:
    python fqs_selector.py --k 2 --nqs --input_path data.json

Usage Random:
    python fqs_selector.py --k 2 --random --seed 42 --input_path data.json

Usage with No Sort (Keep algorithm's output order instead of original order):
    python fqs_selector.py --k 2 --input_path data.json --no_sort
"""

import numpy as np
import torch
import json
import argparse
import os
import random
from tqdm import tqdm


def load_clip_model():
    """Load CLIP model for text encoding."""
    try:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"Loaded CLIP ViT-B/32 on {device}")
        return model, device
    except ImportError:
        print("CLIP not found. Installing...")
        os.system("pip install git+https://github.com/openai/CLIP.git")
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, device


def encode_texts(texts, model, device, batch_size=32):
    """Encode texts into embeddings using CLIP."""
    import clip
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(embeddings)


def compute_distance_matrix(query_embeddings):
    """Compute pairwise distance matrix between query embeddings."""
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.cpu().numpy()
    
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized = query_embeddings / (norms + 1e-8)
    
    similarity = np.dot(normalized, normalized.T)
    distance = 1 - similarity
    
    return distance, similarity


def farthest_query_selection(query_embeddings, k=2, threshold=None, return_indices=True):
    """Farthest Query Selection (FQS): Maximizes the minimum distance."""
    if isinstance(query_embeddings, torch.Tensor):
        embeddings_np = query_embeddings.cpu().numpy()
    else:
        embeddings_np = query_embeddings.copy()

    n_queries = embeddings_np.shape[0]
    if n_queries < k + 1:
        raise ValueError(f"Need at least {k+1} queries, but got {n_queries}")

    distance_matrix, similarity_matrix = compute_distance_matrix(embeddings_np)
    selected_indices = [0]
    remaining_indices = list(range(1, n_queries))

    for _ in range(k):
        valid_candidates = remaining_indices
        if threshold is not None:
            valid_candidates = [idx for idx in remaining_indices if similarity_matrix[0, idx] >= threshold]

        if not valid_candidates:
            selected_indices.append(0)
            continue

        max_min_distance = -1.0
        farthest_idx = -1

        for idx in valid_candidates:
            distances_to_selected = [distance_matrix[idx, s_idx] for s_idx in selected_indices]
            min_distance = min(distances_to_selected)

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = idx

        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)

    return selected_indices if return_indices else embeddings_np[selected_indices]


def nearest_query_sampling(query_embeddings, k=2, threshold=None, return_indices=True):
    """Nearest Query Sampling (NQS): Minimizes the minimum distance."""
    if isinstance(query_embeddings, torch.Tensor):
        embeddings_np = query_embeddings.cpu().numpy()
    else:
        embeddings_np = query_embeddings.copy()

    n_queries = embeddings_np.shape[0]
    if n_queries < k + 1:
        raise ValueError(f"Need at least {k+1} queries, but got {n_queries}")

    distance_matrix, similarity_matrix = compute_distance_matrix(embeddings_np)
    selected_indices = [0]
    remaining_indices = list(range(1, n_queries))

    for _ in range(k):
        valid_candidates = remaining_indices
        if threshold is not None:
            valid_candidates = [idx for idx in remaining_indices if similarity_matrix[0, idx] >= threshold]

        if not valid_candidates:
            selected_indices.append(0)
            continue

        min_min_distance = float('inf')
        nearest_idx = -1

        for idx in valid_candidates:
            distances_to_selected = [distance_matrix[idx, s_idx] for s_idx in selected_indices]
            min_distance = min(distances_to_selected)

            if min_distance < min_min_distance:
                min_min_distance = min_distance
                nearest_idx = idx

        selected_indices.append(nearest_idx)
        remaining_indices.remove(nearest_idx)

    return selected_indices if return_indices else embeddings_np[selected_indices]


def apply_selection_to_json_item(original, augment, k, model=None, device=None, threshold=None, is_nqs=False, is_random=False, sort_results=True):
    """Apply selected algorithm to select k augmentations for a single caption group."""
    if len(augment) < k:
        print(f"Warning: Only {len(augment)} augments available, need {k}. Returning all.")
        return augment

    # Logic chọn k augment
    if is_random:
        selected_augments = random.sample(augment, k)
    else:
        # FQS/NQS Logic - Requires CLIP embeddings
        all_texts = [original] + augment
        embeddings = encode_texts(all_texts, model, device)

        if is_nqs:
            selected_indices = nearest_query_sampling(embeddings, k=k, threshold=threshold, return_indices=True)
        else:
            selected_indices = farthest_query_selection(embeddings, k=k, threshold=threshold, return_indices=True)

        selected_augments = [all_texts[idx] for idx in selected_indices[1:]]

    # --- PHẦN MỚI THÊM: SẮP XẾP LẠI THEO THỨ TỰ GỐC ---
    if sort_results:
        # Tạo mapping để tra cứu vị trí (index) nhanh của câu augment trong mảng gốc
        original_order = {text: idx for idx, text in enumerate(augment)}
        # Sắp xếp mảng kết quả: dùng .get(x, -1) để tránh KeyError nếu x là câu gốc (fallback do threshold)
        selected_augments.sort(key=lambda x: original_order.get(x, -1))

    return selected_augments


def main():
    parser = argparse.ArgumentParser(
        description="Apply Query Selection (FQS, NQS, or Random) to augmented JSON captions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--k", type=int, default=2, help="Number of augment captions to select per video")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--threshold", "-s", type=float, default=None, help="Minimum similarity threshold 's'")
    
    # Algorithm flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nqs", action="store_true", help="Use Nearest Query Sampling (NQS)")
    group.add_argument("--random", action="store_true", help="Use Random Query Sampling")
    
    # Flags for sorting
    parser.add_argument("--no_sort", action="store_true", help="If passed, will NOT sort the output back to original JSON order.")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed if using random sampling
    if args.random:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        if args.random:
            algo_prefix = "random"
        elif args.nqs:
            algo_prefix = "nqs"
        else:
            algo_prefix = "fqs"
        args.output_path = os.path.join(input_dir, f"{base_name}_{algo_prefix}_k_{args.k}.json")
    
    if args.random:
        algo_name = f"RANDOM QUERY SAMPLING (Seed={args.seed})"
    elif args.nqs:
        algo_name = "NEAREST QUERY SAMPLING (NQS)"
    else:
        algo_name = "FARTHEST QUERY SELECTION (FQS)"
    
    print("=" * 70)
    print(algo_name)
    print("=" * 70)
    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"k:      {args.k} (augmentations selected per caption)")
    print(f"Sort:   {'No' if args.no_sort else 'Yes'} (Restore original JSON order)")
    if not args.random:
        print(f"Threshold (s): {args.threshold if args.threshold is not None else 'None'}")
    print("=" * 70)
    
    # Only load CLIP if we actually need it (FQS or NQS)
    model, device = None, None
    if not args.random:
        print("\nLoading CLIP model...")
        model, device = load_clip_model()
    else:
        print("\nSkipping CLIP loading (Random mode is fast!)...")
    
    # Đọc dữ liệu JSON
    print(f"\nLoading JSON from: {args.input_path}")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"Loaded {len(input_data)} videos")
    
    print(f"\nApplying Algorithm (k={args.k})...")
    output_data = {}
    
    # Sort behavior config
    sort_results = not args.no_sort

    # Xử lý từng video và từng caption
    for video_id, caps in tqdm(input_data.items(), desc="Processing videos"):
        output_data[video_id] = {}
        for cap_key, cap_data in caps.items():
            original_text = cap_data["original"]
            augment_texts = cap_data["augment"]
            
            # Lựa chọn k câu augment (bao gồm truyền flag sort_results)
            selected_augments = apply_selection_to_json_item(
                original=original_text,
                augment=augment_texts,
                k=args.k,
                model=model,
                device=device,
                threshold=args.threshold,
                is_nqs=args.nqs,
                is_random=args.random,
                sort_results=sort_results
            )
            
            output_data[video_id][cap_key] = {
                "original": original_text,
                "augment": selected_augments
            }
    
    print(f"\nSaving results...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    algo_short = "RANDOM" if args.random else ("NQS" if args.nqs else "FQS")
    print(f"{algo_short} COMPLETED!")
    print("=" * 70)
    print(f"Processed: {len(input_data)} videos")
    print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    main()