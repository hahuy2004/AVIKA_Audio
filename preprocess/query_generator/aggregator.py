"""
Aggregator for combining multiple similarity matrices from enriched queries.

Four strategies:
1. Weighted RRF: Weighted Reciprocal Rank Fusion
2. Average Similarity: Simple average of similarity matrices
3. True Majority Voting: Hard voting with tie-breaking by original query similarity
4. Max Similarity: Max pooling over query variants
"""

import numpy as np


class Aggregator:
    """
    Aggregates multiple similarity matrices using different strategies.
    
    Strategies:
        1 = Weighted RRF (Weighted Reciprocal Rank Fusion)
        2 = Average Similarity (simple average of similarity matrices)
        3 = True Majority Voting (hard voting with tie-breaking)
        4 = Max Similarity (max pooling over query variants)
    """
    
    def __init__(self, strategy=1):
        """
        Initialize Aggregator.
        
        Args:
            strategy (int): Aggregation strategy
                           1 = Weighted RRF (Reciprocal Rank Fusion)
                           2 = Average Similarity
                           3 = True Majority Voting (hard voting)
                           4 = Max Similarity (max pooling)
        """
        if strategy not in [1, 2, 3, 4]:
            raise ValueError(
                f"Invalid strategy: {strategy}. Must be 1 (Weighted RRF), 2 (Average), 3 (True Majority Voting), or 4 (Max Similarity)"
            )
        
        self.strategy = strategy
        if strategy == 1:
            self.strategy_name = "Weighted RRF"
        elif strategy == 2:
            self.strategy_name = "Average Similarity"
        elif strategy == 3:
            self.strategy_name = "True Majority Voting"
        else:
            self.strategy_name = "Max Similarity"
    
    def aggregate(self, sim_matrices):
        """
        Aggregate multiple similarity matrices.
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
                                      k+1 similarity matrices to aggregate
        
        Returns:
            np.ndarray: Aggregated similarity matrix, shape (n_queries, n_videos)
        """
        if self.strategy == 1:
            return self.weighted_rrf_aggregation(sim_matrices)
        elif self.strategy == 2:
            return self.average_aggregation(sim_matrices)
        elif self.strategy == 3:
            return self.true_majority_voting(sim_matrices)
        else:  # strategy == 4
            return self.max_aggregation(sim_matrices)

    def max_aggregation(self, sim_matrices):
        """
        Max Similarity aggregation.

        Takes the highest logit score for each (query, video) pair across k+1 query variants.
        This helps preserve a strong match when some enriched queries are noisy.

        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)

        Returns:
            np.ndarray: Max-pooled similarity matrix, shape (n_queries, n_videos)
        """
        return np.max(sim_matrices, axis=0)
    
    def average_aggregation(self, sim_matrices):
        """
        Average Similarity aggregation.
        
        Simply computes the arithmetic mean of all similarity matrices.
        
        Formula:
            Sim_final = (Sim_0 + Sim_1 + ... + Sim_k) / (k + 1)
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
        
        Returns:
            np.ndarray: Averaged similarity matrix, shape (n_queries, n_videos)
        """
        # Simple arithmetic mean along axis 0
        avg_sim = np.mean(sim_matrices, axis=0)
        return avg_sim
    
    def weighted_rrf_aggregation(self, sim_matrices):
        """
        Weighted Reciprocal Rank Fusion (RRF) aggregation.
        
        Improved Majority Voting aggregation using Weighted Reciprocal Rank Fusion (RRF).
        
        Key Improvements over naive 1/rank:
        1. Weighted voting: Original query has higher weight than enriched queries
        2. Smoothing constant k: Reduces harshness of pure 1/rank formula
        3. No normalization: Preserves relative magnitudes across queries
        
        Process:
        1. Assign weights: Original query = 1.0, Enriched queries = 0.4
        2. For each similarity matrix, convert to rankings (0-based)
        3. Apply weighted RRF formula: w * (1 / (k + rank + 1))
        4. Sum weighted scores across all query variants
        
        Formula:
            Score(q, v) = Σ_i [ w_i * (1 / (k + Rank_i(q, v) + 1)) ]
        where:
            w_i = weight of i-th query variant (original=1.0, enriched=0.4)
            k = smoothing constant (1.0)
            Rank_i(q, v) = 0-based rank of video v for query q in variant i
        
        Why this works:
        - Weighted: Original query (ground truth) has 2.5x influence vs enriched
        - Smoothing k=1.0: Rank 1→0.5, Rank 2→0.33 (33% drop vs 50% in pure 1/rank)
        - No normalization: Fair comparison across different queries
        
        Example (k=1.0, weights=[1.0, 0.4, 0.4]):
            Video A (correct):
                Original: Rank 0 → 1.0 * 1/(1+0+1) = 0.50
                Enriched1: Rank 2 → 0.4 * 1/(1+2+1) = 0.10
                Enriched2: Rank 2 → 0.4 * 1/(1+2+1) = 0.10
                Total: 0.70
            
            Video B (incorrect):
                Original: Rank 1 → 1.0 * 1/(1+1+1) = 0.33
                Enriched1: Rank 0 → 0.4 * 1/(1+0+1) = 0.20
                Enriched2: Rank 1 → 0.4 * 1/(1+1+1) = 0.13
                Total: 0.66
            
            → Video A wins! Original query's signal is preserved.
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
                                      k+1 similarity matrices from query variants
        
        Returns:
            np.ndarray: Aggregated similarity matrix, shape (n_queries, n_videos)
        """
        k_plus_1, n_queries, n_videos = sim_matrices.shape
        
        # ---------------------------------------------------------
        # IMPROVED AGGREGATION LOGIC (Weighted RRF)
        # ---------------------------------------------------------
        
        # 1. Configure weights for query variants
        weights = [1.0]
        
        # Lựa chọn 1: Trọng số giảm dần theo điều hòa (harmonic decay)
        # Original query keeps full weight; enriched queries decay with index
        # to prevent the sum of enriched weights from overwhelming the original
        # when k is large.
        for i in range(1, k_plus_1):
            # Trọng số giảm dần điều hòa: w_i = 0.5 / i
            # i=1 → 0.500, i=2 → 0.250, i=3 → 0.167, i=4 → 0.125, ...
            weights.append(0.5 / i)
            # Hoặc có thể sử dụng cách gán trọng số giảm dần khác tùy ý

        # # Lựa chọn 2: Trọng số cố định cho enriched queries
        # Sẽ có TH tổng trọng số enriched query lấn át original nếu k lớn
        # Tổng trọng số enriched query có thể lớn hơn original nhưng nên giữ ở mức vừa phải 
        # để không lấn át hoàn toàn tín hiệu từ original query (Khoảng 0.2-0.5 mỗi enriched query sẽ ổn)
        # if k_plus_1 > 1:
        #     # Chia sẻ trọng số (Shared Weight)
        #     # Câu gốc luôn có trọng số là 1
        #     # Câu enriched chia đều phần còn lại của trọng số được cho
        #     if k_plus_1 == 3:
        #         # 2 enriched queries → mỗi câu 0.5
        #         aug_weight = 1.0 / (k_plus_1 - 1)
        #     elif k_plus_1 == 7:
        #         # 6 enriched queries → mỗi câu 0.25
        #         aug_weight = 1.5 / (k_plus_1 - 1)
        #     elif k_plus_1 == 11:
        #         # 10 enriched queries → mỗi câu 0.2
        #         aug_weight = 2.0 / (k_plus_1 - 1)
        #     else:
        #         # Gán trọng số mong muốn cho enriched queries (ví dụ: 0.4)
            
        #     # Gán trọng số bằng nhau cho tất cả các câu từ index 1 trở đi
        #     for i in range(1, k_plus_1):
        #         weights.append(aug_weight)
        
        # 2. Smoothing constant for RRF
        # k=1.0 provides good balance:
        #   - Rank 1: 1/(1+1) = 0.50
        #   - Rank 2: 1/(1+2) = 0.33 (33% drop, gentler than 50%)
        #   - Rank 10: 1/(1+10) = 0.09
        # Smaller k → steeper curve (better R@1)
        # Larger k → flatter curve (better R@5, R@10)
        k_smooth = 1.0
        
        # Initialize final score matrix
        final_score_matrix = np.zeros((n_queries, n_videos))
        
        # 3. Process each query variant with its weight
        for idx, sim_matrix in enumerate(sim_matrices):
            w = weights[idx]
            
            # Get 0-based ranks for each query-video pair
            # argsort(-sim_matrix, axis=1): sort descending (highest sim first)
            # argsort again: convert sorted indices to ranks (0-based)
            # Result: ranks[i, j] = rank of video j for query i (0=best)
            ranks = np.argsort(np.argsort(-sim_matrix, axis=1), axis=1)
            
            # Apply weighted RRF formula: w * (1 / (k + rank + 1))
            # rank + 1: convert 0-based to 1-based for formula
            score = w * (1.0 / (k_smooth + ranks + 1))
            
            # Accumulate weighted scores
            final_score_matrix += score
        
        # 4. No normalization needed
        # compute_metrics() only cares about relative ranking order,
        # not absolute score magnitudes. Normalization can introduce bias
        # across different queries.
        
        return final_score_matrix
    
    def true_majority_voting(self, sim_matrices):
        """
        True Majority Voting aggregation with tie-breaking.
        
        Each query variant "votes" for its top-1 video. The video with the most votes wins.
        In case of ties (e.g., 1-1-1), we use the original query's similarity as a tie-breaker.
        
        Why this handles 1-1-1 ties:
        - Original query votes for Video A (vote=1)
        - Enriched query 1 votes for Video B (vote=1)
        - Enriched query 2 votes for Video C (vote=1)
        
        Tie-breaking formula:
            Final_Score = Vote_Count + (1e-4 * Similarity_From_Original_Query)
        
        Since Video A is top-1 for the original query, Sim(Original, A) > Sim(Original, B).
        Therefore, Video A gets the highest final score and wins.
        
        Process:
        1. For each query variant, find its top-1 video (argmax)
        2. Count votes for each video
        3. Add tiny weighted similarity from original query as tie-breaker
        4. Return final score matrix
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
        
        Returns:
            np.ndarray: Final score matrix, shape (n_queries, n_videos)
        """
        k_plus_1, n_queries, n_videos = sim_matrices.shape
        
        # Initialize vote count matrix and final score matrix
        vote_counts = np.zeros((n_queries, n_videos))
        
        # Step 1: Count votes from all query variants
        for idx in range(k_plus_1):
            sim_matrix = sim_matrices[idx]  # Shape: (n_queries, n_videos)
            
            # For each query, find its top-1 video
            top1_videos = np.argmax(sim_matrix, axis=1)  # Shape: (n_queries,)
            
            # Cast votes using vectorized indexing (equivalent to loop but faster)
            vote_counts[np.arange(n_queries), top1_videos] += 1
        
        # Step 2: Add tie-breaker using original query's similarity
        # Formula: Final_Score = Vote_Count + (1e-4 * Sim_Original)
        # The 1e-4 weight ensures votes dominate, but ties are broken by original similarity
        sim_original = sim_matrices[0]  # Original query's similarity matrix
        tie_breaker_weight = 1e-4
        
        final_score_matrix = vote_counts + (tie_breaker_weight * sim_original)
        
        # Step 3: Return final scores
        # Note: Higher score = better, same as similarity matrices
        return final_score_matrix


def test_aggregator():
    """Test the Aggregator class with dummy data and verify weighted RRF logic."""
    print("="*70)
    print("TESTING IMPROVED AGGREGATOR (Weighted RRF)")
    print("="*70)
    
    np.random.seed(42)
    
    # Create dummy similarity matrices
    k_plus_1 = 3  # 1 original + 2 enriched = 3 total queries per video
    n_queries = 10  # 10 test queries
    n_videos = 100  # 100 candidate videos
    
    # Generate k+1 similarity matrices
    sim_matrices = np.random.rand(k_plus_1, n_queries, n_videos)
    
    print(f"\nInput: {k_plus_1} similarity matrices")
    print(f"Shape: ({k_plus_1}, {n_queries}, {n_videos})")
    print(f"  - {k_plus_1} retrieval runs (1 original + {k_plus_1-1} enriched)")
    print(f"  - {n_queries} test queries")
    print(f"  - {n_videos} candidate videos")
    print(f"\nWeights: [1.0 (original), 0.4 (enriched1), 0.4 (enriched2)]")
    print(f"Smoothing constant k: 1.0")
    
    # Test Strategy 1: Weighted RRF
    print("\n" + "-"*70)
    print("Testing Strategy 1: Weighted RRF")
    print("-"*70)
    
    aggregator_rrf = Aggregator(strategy=1)
    final_sim_rrf = aggregator_rrf.aggregate(sim_matrices)
    
    print(f"Output shape: {final_sim_rrf.shape}")
    print(f"Output range: [{final_sim_rrf.min():.4f}, {final_sim_rrf.max():.4f}]")
    
    # Show top-5 videos for first query
    query_0_scores_rrf = final_sim_rrf[0]
    top5_indices_rrf = np.argsort(query_0_scores_rrf)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (Weighted RRF):")
    for rank, video_idx in enumerate(top5_indices_rrf, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores_rrf[video_idx]:.4f})")
    
    # Verify weighted contribution
    print(f"\nVerifying weighted RRF formula for Query 0, Video {top5_indices_rrf[0]}:")
    vid_idx = top5_indices_rrf[0]
    for k_idx in range(k_plus_1):
        sim = sim_matrices[k_idx, 0, vid_idx]
        rank = np.sum(sim_matrices[k_idx, 0, :] > sim)  # 0-based rank
        weight = 1.0 if k_idx == 0 else 0.4
        contribution = weight * (1.0 / (1.0 + rank + 1))
        variant_name = "Original" if k_idx == 0 else f"Enriched{k_idx}"
        print(f"  {variant_name}: rank={rank}, weight={weight:.1f}, contribution={contribution:.4f}")
    
    # Test Strategy 2: Average Similarity
    print("\n" + "-"*70)
    print("Testing Strategy 2: Average Similarity")
    print("-"*70)
    
    aggregator_avg = Aggregator(strategy=2)
    final_sim_avg = aggregator_avg.aggregate(sim_matrices)
    
    print(f"Output shape: {final_sim_avg.shape}")
    print(f"Output range: [{final_sim_avg.min():.4f}, {final_sim_avg.max():.4f}]")
    
    # Show top-5 videos for first query
    query_0_scores_avg = final_sim_avg[0]
    top5_indices_avg = np.argsort(query_0_scores_avg)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (Average):")
    for rank, video_idx in enumerate(top5_indices_avg, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores_avg[video_idx]:.4f})")
    
    # Test Strategy 3: True Majority Voting
    print("\n" + "-"*70)
    print("Testing Strategy 3: True Majority Voting (Hard Voting)")
    print("-"*70)
    
    aggregator_majority = Aggregator(strategy=3)
    final_sim_majority = aggregator_majority.aggregate(sim_matrices)
    
    print(f"Output shape: {final_sim_majority.shape}")
    print(f"Output range: [{final_sim_majority.min():.4f}, {final_sim_majority.max():.4f}]")
    
    # Show top-5 videos for first query
    query_0_scores_majority = final_sim_majority[0]
    top5_indices_majority = np.argsort(query_0_scores_majority)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (True Majority Voting):")
    for rank, video_idx in enumerate(top5_indices_majority, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores_majority[video_idx]:.4f})")

    # Test Strategy 4: Max Similarity
    print("\n" + "-"*70)
    print("Testing Strategy 4: Max Similarity")
    print("-"*70)

    aggregator_max = Aggregator(strategy=4)
    final_sim_max = aggregator_max.aggregate(sim_matrices)

    print(f"Output shape: {final_sim_max.shape}")
    print(f"Output range: [{final_sim_max.min():.4f}, {final_sim_max.max():.4f}]")

    query_0_scores_max = final_sim_max[0]
    top5_indices_max = np.argsort(query_0_scores_max)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (Max Similarity):")
    for rank, video_idx in enumerate(top5_indices_max, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores_max[video_idx]:.4f})")
    
    # Show voting details for top-1 video
    print(f"\nVoting details for Query 0, Top-1 Video {top5_indices_majority[0]}:")
    top1_vid = top5_indices_majority[0]
    for k_idx in range(k_plus_1):
        top1_of_variant = np.argmax(sim_matrices[k_idx, 0, :])
        voted_for_top1 = "✓" if top1_of_variant == top1_vid else "✗"
        variant_name = "Original" if k_idx == 0 else f"Enriched{k_idx}"
        print(f"  {variant_name}: top-1 = Video {top1_of_variant} {voted_for_top1}")
    
    # Compare strategies
    print("\n" + "="*70)
    print("COMPARISON BETWEEN ALL STRATEGIES")
    print("="*70)
    
    print(f"\nStrategy 1 (Weighted RRF) vs Strategy 2 (Average):")
    diff_rrf_avg = np.abs(final_sim_rrf - final_sim_avg)
    print(f"  Mean absolute difference: {diff_rrf_avg.mean():.4f}")
    print(f"  Max absolute difference: {diff_rrf_avg.max():.4f}")
    
    print(f"\nStrategy 1 (Weighted RRF) vs Strategy 3 (True Majority Voting):")
    diff_rrf_maj = np.abs(final_sim_rrf - final_sim_majority)
    print(f"  Mean absolute difference: {diff_rrf_maj.mean():.4f}")
    print(f"  Max absolute difference: {diff_rrf_maj.max():.4f}")
    
    print(f"\nStrategy 2 (Average) vs Strategy 3 (True Majority Voting):")
    diff_avg_maj = np.abs(final_sim_avg - final_sim_majority)
    print(f"  Mean absolute difference: {diff_avg_maj.mean():.4f}")
    print(f"  Max absolute difference: {diff_avg_maj.max():.4f}")

    print(f"\nStrategy 2 (Average) vs Strategy 4 (Max Similarity):")
    diff_avg_max = np.abs(final_sim_avg - final_sim_max)
    print(f"  Mean absolute difference: {diff_avg_max.mean():.4f}")
    print(f"  Max absolute difference: {diff_avg_max.max():.4f}")
    
    # Agreement on top-1
    print(f"\nTop-1 agreement across strategies:")
    agreement_rrf_avg = 0
    agreement_rrf_maj = 0
    agreement_avg_maj = 0
    agreement_rrf_max = 0
    
    for q_idx in range(n_queries):
        top1_rrf = np.argmax(final_sim_rrf[q_idx])
        top1_avg = np.argmax(final_sim_avg[q_idx])
        top1_maj = np.argmax(final_sim_majority[q_idx])
        
        if top1_rrf == top1_avg:
            agreement_rrf_avg += 1
        if top1_rrf == top1_maj:
            agreement_rrf_maj += 1
        if top1_avg == top1_maj:
            agreement_avg_maj += 1
        if top1_rrf == np.argmax(final_sim_max[q_idx]):
            agreement_rrf_max += 1
    
    print(f"  RRF vs Average: {agreement_rrf_avg}/{n_queries} ({100.0*agreement_rrf_avg/n_queries:.1f}%)")
    print(f"  RRF vs Majority: {agreement_rrf_maj}/{n_queries} ({100.0*agreement_rrf_maj/n_queries:.1f}%)")
    print(f"  Average vs Majority: {agreement_avg_maj}/{n_queries} ({100.0*agreement_avg_maj/n_queries:.1f}%)")
    print(f"  RRF vs Max: {agreement_rrf_max}/{n_queries} ({100.0*agreement_rrf_max/n_queries:.1f}%)")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    test_aggregator()
