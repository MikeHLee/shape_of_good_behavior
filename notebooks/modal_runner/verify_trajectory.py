import pandas as pd
import numpy as np

def verify_trajectory_shift():
    try:
        df = pd.read_parquet("results/comparative_analysis.parquet")
        print(f"Loaded {len(df)} rows")
        
        # Ensure we have risk scores
        if "prompt_harmonic_risk" not in df.columns:
            print("Error: 'prompt_harmonic_risk' column missing")
            return

        # Get unique prompts and their risk
        prompt_risks = df[["prompt_id", "prompt_harmonic_risk"]].drop_duplicates()
        
        # Identify top 5% riskiest prompts
        threshold = prompt_risks["prompt_harmonic_risk"].quantile(0.95)
        high_risk_ids = prompt_risks[prompt_risks["prompt_harmonic_risk"] >= threshold]["prompt_id"]
        
        print(f"Top 5% risk threshold: {threshold:.4f}")
        print(f"Number of high risk prompts: {len(high_risk_ids)}")
        
        # Filter main df
        high_risk_df = df[df["prompt_id"].isin(high_risk_ids)]
        
        # Compare SGPO vs PPO trajectory shift
        gpo_shifts = high_risk_df[high_risk_df["model"] == "gpo"]["trajectory_shift"]
        ppo_shifts = high_risk_df[high_risk_df["model"] == "ppo"]["trajectory_shift"]
        
        print(f"SGPO Mean Shift (High Risk): {gpo_shifts.mean():.4f}")
        print(f"PPO Mean Shift (High Risk): {ppo_shifts.mean():.4f}")
        
        # Check claim: "GeoDPO achieves a positive trajectory shift for 50% of cases"
        # Interpreting as: How often is SGPO shift > PPO shift?
        
        # We need to align by prompt_id
        gpo_df = high_risk_df[high_risk_df["model"] == "gpo"].set_index("prompt_id")["trajectory_shift"]
        ppo_df = high_risk_df[high_risk_df["model"] == "ppo"].set_index("prompt_id")["trajectory_shift"]
        
        # Join
        comparison = pd.concat([gpo_df, ppo_df], axis=1, keys=["gpo", "ppo"]).dropna()
        
        if len(comparison) == 0:
            print("No overlapping prompts found between SGPO and PPO in high risk set")
            return

        gpo_greater = (comparison["gpo"] > comparison["ppo"]).mean()
        print(f"Fraction of cases where SGPO shift > PPO shift: {gpo_greater:.2%}")
        
        # Check magnitude of difference
        diff = comparison["gpo"] - comparison["ppo"]
        print(f"Mean difference (SGPO - PPO): {diff.mean():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_trajectory_shift()
