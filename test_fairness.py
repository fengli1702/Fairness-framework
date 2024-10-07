import pandas as pd
from scipy.stats import kendalltau, spearmanr

def calculate_ranking_discrepancy(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Get unique origin_ids
    origin_ids = df['origin_id'].unique()

    discrepancies = []

    for origin_id in origin_ids:
        # Filter data for each origin_id group
        group = df[df['origin_id'] == origin_id]

        # Sort by user_id (original ranking)
        group = group.sort_values(by='user_id')

        # Get original and theta-based rankings
        original_ranking = group['user_id'].rank(method='first').values
        theta_ranking = group['theta'].rank(ascending=False, method='first').values

        # Calculate Kendall's Tau and Spearman's rank correlation
        kendall_tau, _ = kendalltau(original_ranking, theta_ranking)
        spearman_corr, _ = spearmanr(original_ranking, theta_ranking)

        discrepancies.append({
            'origin_id': origin_id,
            'kendall_tau': kendall_tau,
            'spearman_corr': spearman_corr
        })

    # Create a dataframe with results
    result_df = pd.DataFrame(discrepancies)
    print(result_df)

    return result_df

if __name__ == "__main__":
    data_path = "./v_ability_parameters.csv"
    result_df = calculate_ranking_discrepancy(data_path)
    result_df.to_csv("ranking_discrepancy_results.csv", index=False)
    print("Results saved to ranking_discrepancy_results.csv")
