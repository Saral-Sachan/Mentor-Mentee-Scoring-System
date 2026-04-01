import pandas as pd
import numpy as np
import ast

def load_data():
    # Load the raw data
    mentors_df = pd.read_csv('mentors.csv')
    students_df = pd.read_csv('students.csv')
    interactions_df = pd.read_csv('interactions.csv')
    
    return mentors_df, students_df, interactions_df


def clean_and_map_data(mentors_df, students_df):
    # 1. Manually parse the list string instead of using ast.literal_eval
    # This strips the '[' and ']', splits by commas, and removes extra spaces or quotes.
    mentors_df['Projects'] = mentors_df['Projects'].apply(
        lambda x: [item.strip().strip("'\"") for item in x.strip('[]').split(',')] if isinstance(x, str) else x
    )
    
    # 2. Explode the mentors dataframe so each project has its own row
    mentor_project_mapping = mentors_df.explode('Projects').rename(columns={'Projects': 'ProjectID'})
    
    # 3. Map students to their mentors based on the ProjectID
    student_mentor_df = pd.merge(students_df, mentor_project_mapping[['MentorID', 'ProjectID']], on='ProjectID', how='left')
    
    return mentors_df, student_mentor_df



def aggregate_mentor_stats(mentors_df, student_mentor_df, interactions_df):
    # 1. Calculate how many mentees each mentor has (N)
    mentee_counts = student_mentor_df.groupby('MentorID')['StudentID'].nunique().reset_index(name='TotalMentees')
    
    # 2. Calculate the Progress metric (P)
    # P = Sum of completed milestones / Sum of total milestones for that mentor's cohort
    progress_grouped = student_mentor_df.groupby('MentorID').agg(
        Total_Completed=('MilestonesCompleted', 'sum'),
        Total_Assigned=('TotalMilestones', 'sum')
    ).reset_index()
    progress_grouped['Raw_Progress'] = progress_grouped['Total_Completed'] / progress_grouped['Total_Assigned']
    
    # 3. Aggregate Interactions (Meetings, Code Reviews, Messages, and Response Time)
    # We sum the interaction counts and take the mean of the response times
    interactions_grouped = interactions_df.groupby('MentorID').agg(
        Total_Meetings=('Meetings', 'sum'),
        Total_CodeReviews=('CodeReviews', 'sum'),
        Total_Messages=('Messages', 'sum'),
        Avg_ResponseTime=('AvgResponseTime', 'mean')
    ).reset_index()
    
    # 4. Merge everything into a master stats dataframe
    master_stats = mentors_df[['MentorID', 'Name', 'Domain']].copy()
    master_stats = pd.merge(master_stats, mentee_counts, on='MentorID', how='left')
    master_stats = pd.merge(master_stats, progress_grouped[['MentorID', 'Raw_Progress']], on='MentorID', how='left')
    master_stats = pd.merge(master_stats, interactions_grouped, on='MentorID', how='left')
    
    # Fill any NaNs with 0 (e.g., if a mentor had 0 interactions)
    master_stats.fillna(0, inplace=True)
    
    return master_stats


import numpy as np

def calculate_core_metrics(master_stats, feedback_df=None):
    # 1. Progress Score (P)
    # Already calculated as 'Raw_Progress' in the aggregation step.
    master_stats['P_Score'] = master_stats['Raw_Progress']
    
    # 2. Responsiveness Score (R)
    # Using the Rational Decay function: 1 / (1 + (t_avg / t_half)^2)
    t_half = 24 # 24 hours is our standard for a 0.5 score
    master_stats['R_Score'] = 1 / (1 + (master_stats['Avg_ResponseTime'] / t_half)**2)
    
    # 3. Engagement Score (E)
    # E_raw = (5*CR + 3*Meetings + 0.1*Messages) / TotalMentees
    # To avoid division by zero if a mentor has 0 mentees, we use np.where
    master_stats['E_raw'] = np.where(
        master_stats['TotalMentees'] > 0,
        (5 * master_stats['Total_CodeReviews'] + 
         3 * master_stats['Total_Meetings'] + 
         0.1 * master_stats['Total_Messages']) / master_stats['TotalMentees'],
        0
    )
    # Bound it between 0 and 1 using tanh
    e_target = 10 # Baseline healthy engagement points per student
    master_stats['E_Score'] = np.tanh(master_stats['E_raw'] / e_target)
    
    # 4. Mentee Feedback Score (F)
    # If you have feedback data, apply the Bayesian Average
    if feedback_df is not None and not feedback_df.empty:
        # Assuming feedback_df has columns: ['MentorID', 'Rating']
        feedback_stats = feedback_df.groupby('MentorID').agg(
            Avg_Rating=('Rating', 'mean'),
            Num_Ratings=('Rating', 'count')
        ).reset_index()
        
        global_mean = feedback_df['Rating'].mean()
        m = 3 # Confidence threshold (need ~3 reviews to trust the average)
        
        # Merge back to master
        master_stats = pd.merge(master_stats, feedback_stats, on='MentorID', how='left')
        master_stats['Avg_Rating'].fillna(global_mean, inplace=True)
        master_stats['Num_Ratings'].fillna(0, inplace=True)
        
        # Calculate Bayesian Average and normalize to [0, 1] by dividing by 5
        master_stats['F_Score'] = (
            (master_stats['Num_Ratings'] / (master_stats['Num_Ratings'] + m)) * master_stats['Avg_Rating'] +
            (m / (master_stats['Num_Ratings'] + m)) * global_mean
        ) / 5.0
    else:
        # Fallback if no data is provided: assign a neutral 0.5 or 1.0 depending on your design choice
        master_stats['F_Score'] = 0.5 

    return master_stats

def update_scores_over_time(current_stats_df, historical_scores_df=None, alpha=0.6, decay_rate=0.15):
    """
    Updates the mentor scores based on historical data.
    alpha: Weight of the current period's score (0 < alpha < 1).
    decay_rate: Penalty for being inactive for 2 consecutive periods.
    """
    # If this is the first evaluation period, the current scores are the final scores.
    if historical_scores_df is None or historical_scores_df.empty:
        current_stats_df['Final_M_Score'] = current_stats_df['Current_M_Score']
        return current_stats_df

    # Merge current stats with historical stats
    # Assuming historical_scores_df has ['MentorID', 'Past_M_Score', 'Inactive_Periods']
    merged_df = pd.merge(current_stats_df, historical_scores_df, on='MentorID', how='left')
    
    # Fill missing historical data for new mentors
    merged_df['Past_M_Score'].fillna(merged_df['Current_M_Score'], inplace=True)
    merged_df['Inactive_Periods'].fillna(0, inplace=True)
    
    # Define an 'Active' threshold (e.g., did they have any interactions this period?)
    merged_df['Is_Active'] = (merged_df['Total_Meetings'] + merged_df['Total_CodeReviews'] + merged_df['Total_Messages']) > 0
    
    # Update Inactive Periods
    merged_df['Inactive_Periods'] = np.where(merged_df['Is_Active'], 0, merged_df['Inactive_Periods'] + 1)
    
    # Calculate New Score
    def calculate_new_score(row):
        if row['Inactive_Periods'] >= 2:
            # Apply Activity Decay: M_new = M_old * (1 - d)
            return row['Past_M_Score'] * (1 - decay_rate)
        else:
            # Apply Exponential Smoothing: M_(t+1) = (1 - alpha)*M_t + alpha*M_current
            return (1 - alpha) * row['Past_M_Score'] + alpha * row['Current_M_Score']
            
    merged_df['Final_M_Score'] = merged_df.apply(calculate_new_score, axis=1)
    
    return merged_df

def generate_final_rankings(master_stats):
    # Apply weights defined in Step 1
    w1, w2, w3, w4 = 0.40, 0.15, 0.30, 0.15
    
    # Ensure weights sum to 1
    assert abs((w1 + w2 + w3 + w4) - 1.0) < 1e-6, "Weights must sum to 1"
    
    # Calculate final Mentor Score M(m)
    master_stats['Current_M_Score'] = (
        w1 * master_stats['P_Score'] +
        w2 * master_stats['R_Score'] +
        w3 * master_stats['E_Score'] +
        w4 * master_stats['F_Score']
    )
    
    # Assuming this is period 1 (no historical data), Current is Final
    master_stats['Final_M_Score'] = master_stats['Current_M_Score']
    
    # Sort descending
    master_stats = master_stats.sort_values(by='Final_M_Score', ascending=False).reset_index(drop=True)
    
    # Assign Rank
    master_stats['Rank'] = master_stats.index + 1
    
    # Select requested output columns
    output_df = master_stats[['MentorID', 'Name', 'Final_M_Score', 'Rank']].copy()
    
    # Rename score column to match PDF requested format
    output_df.rename(columns={'Final_M_Score': 'Final Mentor Score M(m)'}, inplace=True)
    
    # Export to CSV
    output_df.to_csv('mentor_scores.csv', index=False)
    print("Successfully generated mentor_scores.csv!")
    
    return output_df

if __name__ == "__main__":
    print("🚀 Starting WnCC Mentor Evaluation System...")

    try:
        # 1. Load the raw CSV data
        print("Loading data...")
        mentors_df, students_df, interactions_df = load_data()

        # 2. Clean data and map students to mentors
        print("Mapping relationships...")
        mentors_df, student_mentor_df = clean_and_map_data(mentors_df, students_df)

        # 3. Aggregate the raw statistics for each mentor
        print("Aggregating mentor statistics...")
        master_stats = aggregate_mentor_stats(mentors_df, student_mentor_df, interactions_df)

        # 4. Apply the mathematical formulas to calculate P, R, E, and F
        print("Calculating core metrics...")
        # Note: We pass None for feedback_df as it wasn't in the provided CSV list
        master_stats = calculate_core_metrics(master_stats, feedback_df=None)

        # 5. Apply weights and generate the final ranked output
        print("Generating final rankings and exporting to CSV...")
        final_rankings_df = generate_final_rankings(master_stats)

        # Print a quick preview to the console to verify it worked
        print("\n✅ Success! Here are the Top 5 Mentors:")
        print(final_rankings_df.head().to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find the required CSV files. {e}")
        print("Make sure mentors.csv, students.csv, and interactions.csv are in the same folder as this script.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")