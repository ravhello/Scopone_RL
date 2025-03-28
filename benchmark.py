#!/usr/bin/env python3
"""
Checkpoint Benchmark Script for Scopone AI

This script compares different Team 0 checkpoints of the Scopone AI model by having them
play against each other to benchmark their performance.

Usage:
  python benchmark_team0.py --checkpoint_dir checkpoints/ --games 1000
  python benchmark_team0.py --checkpoints checkpoints/model_team0_ep5000.pth checkpoints/model_team0_ep10000.pth
"""

import torch
import numpy as np
import argparse
import os
import time
import re
from tqdm import tqdm
from collections import defaultdict
import itertools
import glob
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Side, Alignment, Font
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl

# Import the required components from the existing code
from environment import ScoponeEnvMA
from main import DQNAgent

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_agent_from_checkpoint(checkpoint_path):
    """Load a Team 0 agent from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Always create a Team 0 agent
    agent = DQNAgent(team_id=0)
    agent.load_checkpoint(checkpoint_path)
    # Set epsilon to 0 for deterministic play during evaluation
    agent.epsilon = 0.0
    return agent

def play_game(agent1, agent2, starting_player=0):
    """
    Play a single game between two Team 0 agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
        starting_player: Which player starts (0-3)
    
    Returns:
        winner: 0 if agent1 won, 1 if agent2 won, -1 if draw
        team_scores: Scores for each agent [agent1_score, agent2_score]
        game_length: Number of moves in the game
    """
    env = ScoponeEnvMA()
    env.current_player = starting_player
    
    done = False
    
    # Track which agent is controlling the current player
    # In even positions (0, 2) of Team 0, agent1 plays. In odd positions (1, 3) of Team 1, agent2 plays.
    agent1_positions = [0, 2]  # Team 0 positions
    agent2_positions = [1, 3]  # Team 1 positions
    
    # Game loop
    while not done:
        current_player = env.current_player
        
        # Choose the right agent based on player position
        if current_player in agent1_positions:
            agent = agent1
        else:
            agent = agent2
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            print("\n[ERROR] No valid actions available!")
            break
        
        # Get observation for current player
        obs = env._get_observation(current_player)
        
        # Agent selects action
        action = agent.pick_action(obs, valid_actions, env)
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
    
    # Extract final score information from team_rewards
    agent1_score = 0.0
    agent2_score = 0.0
    if "team_rewards" in info:
        team_rewards = info["team_rewards"]
        agent1_score = team_rewards[0]  # Team 0 score for agent1
        agent2_score = team_rewards[1]  # Team 1 score for agent2
    
    # Determine winner based on team scores
    winner = 0 if agent1_score > agent2_score else 1 if agent2_score > agent1_score else -1
    
    return winner, [agent1_score, agent2_score], len(env.game_state["history"])

def find_checkpoints(checkpoint_dir, pattern="*team0*ep*.pth"):
    """Find Team 0 checkpoint files in the specified directory."""
    if os.path.isfile(checkpoint_dir):
        return [checkpoint_dir]
    
    checkpoint_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    # Sort checkpoints by episode number when possible
    try:
        checkpoint_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]) 
                             if '_ep' in x else float('inf'))
    except:
        # If sorting by episode fails, sort by modification time
        checkpoint_files.sort(key=os.path.getmtime)
    
    return checkpoint_files

def extract_episode_number(checkpoint_path):
    """Extract episode number from checkpoint filename."""
    # Try to find episode number using regex
    match = re.search(r'_ep(\d+)', os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    
    # If not found, try to extract any number from the filename
    match = re.search(r'(\d+)', os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    
    # If still no number, return a very large number to place at the end
    return float('inf')

def evaluate_checkpoints(checkpoint_paths, num_games=10000):
    """Evaluate Team 0 checkpoints against each other in head-to-head matches."""
    results = {}
    
    # Load all agents
    agents = {}
    for path in checkpoint_paths:
        # Extract episode number for better naming
        episode_num = extract_episode_number(path)
        name = f"ep{episode_num}" if episode_num != float('inf') else os.path.basename(path).replace(".pth", "")
        
        print(f"Loading Team 0 agent from checkpoint: {path} as {name}")
        agents[name] = load_agent_from_checkpoint(path)
    
    # Play games between all pairs of agents
    matchups = list(itertools.combinations(agents.keys(), 2))
    
    for agent1_name, agent2_name in matchups:
        print(f"\nEvaluating: {agent1_name} vs {agent2_name}")
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        
        matchup_key = f"{agent1_name}_vs_{agent2_name}"
        
        results[matchup_key] = {
            "games": num_games,
            "agent1_wins": 0,
            "agent2_wins": 0,
            "draws": 0,
            "agent1_score_total": 0,
            "agent2_score_total": 0,
            "agent1_score_distribution": defaultdict(int),
            "agent2_score_distribution": defaultdict(int),
            "first_starter_wins": 0,
            "game_lengths": []
        }
        
        # Progress bar
        pbar = tqdm(total=num_games)
        
        # Play num_games games with rotation of starting player
        for game_idx in range(num_games):
            # Rotate starting player every game
            starting_player = game_idx % 4
            
            # Play game
            winner, scores, game_length = play_game(agent1, agent2, starting_player)
            
            # Record results
            agent1_score, agent2_score = scores
            
            if winner == 0:  # Agent 1 won
                results[matchup_key]["agent1_wins"] += 1
                # Check if the starting team won
                if starting_player in [0, 2]:  # Team 0 started
                    results[matchup_key]["first_starter_wins"] += 1
            elif winner == 1:  # Agent 2 won
                results[matchup_key]["agent2_wins"] += 1
                # Check if the starting team won
                if starting_player in [1, 3]:  # Team 1 started
                    results[matchup_key]["first_starter_wins"] += 1
            else:
                results[matchup_key]["draws"] += 1
            
            # Record scores
            results[matchup_key]["agent1_score_total"] += agent1_score
            results[matchup_key]["agent2_score_total"] += agent2_score
            
            # Update score distributions
            results[matchup_key]["agent1_score_distribution"][agent1_score] += 1
            results[matchup_key]["agent2_score_distribution"][agent2_score] += 1
            
            # Record game length
            results[matchup_key]["game_lengths"].append(game_length)
            
            # Calculate win rates for progress bar
            agent1_wins = results[matchup_key]["agent1_wins"]
            agent2_wins = results[matchup_key]["agent2_wins"]
            win_percentage1 = (agent1_wins / (game_idx + 1)) * 100
            win_percentage2 = (agent2_wins / (game_idx + 1)) * 100
            
            pbar.update(1)
            pbar.set_description(
                f"{agent1_name}: {agent1_wins} ({win_percentage1:.1f}%), "
                f"{agent2_name}: {agent2_wins} ({win_percentage2:.1f}%)"
            )
        
        pbar.close()
        
        # Calculate win rates and first-starter advantage
        agent1_total_wins = results[matchup_key]["agent1_wins"]
        agent2_total_wins = results[matchup_key]["agent2_wins"]
        total_decided_games = agent1_total_wins + agent2_total_wins
        
        # Calculate first-starter advantage
        first_starter_wins = results[matchup_key]["first_starter_wins"] 
        first_starter_advantage = first_starter_wins / total_decided_games if total_decided_games > 0 else 0
        
        # Calculate average scores
        results[matchup_key]["agent1_avg_score"] = results[matchup_key]["agent1_score_total"] / num_games
        results[matchup_key]["agent2_avg_score"] = results[matchup_key]["agent2_score_total"] / num_games
        
        # Calculate detailed win percentages
        agent1_win_pct = agent1_total_wins / num_games * 100
        agent2_win_pct = agent2_total_wins / num_games * 100
        draw_pct = results[matchup_key]["draws"] / num_games * 100
        
        # Print detailed summary
        print(f"\nResults for {agent1_name} vs {agent2_name}:")
        print(f"  {agent1_name} wins: {agent1_total_wins} ({agent1_win_pct:.1f}%)")
        print(f"  {agent2_name} wins: {agent2_total_wins} ({agent2_win_pct:.1f}%)")
        print(f"  Draws: {results[matchup_key]['draws']} ({draw_pct:.1f}%)")
        print(f"  First-starter advantage: {first_starter_advantage:.2f} (1.0 = 100% advantage, 0.5 = no advantage)")
        print(f"  Average score {agent1_name}: {results[matchup_key]['agent1_avg_score']:.2f}")
        print(f"  Average score {agent2_name}: {results[matchup_key]['agent2_avg_score']:.2f}")
        print(f"  Average game length: {sum(results[matchup_key]['game_lengths'])/len(results[matchup_key]['game_lengths']):.1f} moves")
    
    return results

def generate_excel_comparison(checkpoint_paths, results, output_file):
    """
    Generate an Excel file with comparative results between models.
    Orders models by their training episode numbers.
    """
    # Extract checkpoint names and episode numbers
    checkpoints_info = []
    for path in checkpoint_paths:
        episode = extract_episode_number(path)
        name = f"ep{episode}" if episode != float('inf') else os.path.basename(path).replace(".pth", "")
        checkpoints_info.append((name, episode, path))
    
    # Sort checkpoints by episode number
    checkpoints_info.sort(key=lambda x: x[1])
    
    # Create a DataFrame for win rates
    model_names = [info[0] for info in checkpoints_info]
    win_rate_data = pd.DataFrame(index=model_names, columns=model_names)
    score_diff_data = pd.DataFrame(index=model_names, columns=model_names)
    
    # Fill in the win rate and score difference matrices
    for i, (model1, _, _) in enumerate(checkpoints_info):
        for j, (model2, _, _) in enumerate(checkpoints_info):
            if i == j:
                # Diagonal elements (model vs itself)
                win_rate_data.loc[model1, model2] = "—"
                score_diff_data.loc[model1, model2] = "—"
            else:
                # Check if we have results for this matchup
                matchup_key = f"{model1}_vs_{model2}"
                reverse_matchup_key = f"{model2}_vs_{model1}"
                
                if matchup_key in results:
                    # We have direct results
                    data = results[matchup_key]
                    model1_win_rate = data['agent1_wins'] / data['games'] * 100
                    score_diff = data['agent1_avg_score'] - data['agent2_avg_score']
                    
                    win_rate_data.loc[model1, model2] = f"{model1_win_rate:.1f}%"
                    score_diff_data.loc[model1, model2] = f"{score_diff:.2f}"
                    
                    # Also fill in the reverse matchup
                    model2_win_rate = data['agent2_wins'] / data['games'] * 100
                    
                    win_rate_data.loc[model2, model1] = f"{model2_win_rate:.1f}%"
                    score_diff_data.loc[model2, model1] = f"{-score_diff:.2f}"
                    
                elif reverse_matchup_key in results:
                    # We have results for the reverse matchup
                    data = results[reverse_matchup_key]
                    model2_win_rate = data['agent1_wins'] / data['games'] * 100
                    score_diff = data['agent1_avg_score'] - data['agent2_avg_score']
                    
                    win_rate_data.loc[model2, model1] = f"{model2_win_rate:.1f}%"
                    score_diff_data.loc[model2, model1] = f"{score_diff:.2f}"
                    
                    # Also fill in the forward matchup
                    model1_win_rate = data['agent2_wins'] / data['games'] * 100
                    
                    win_rate_data.loc[model1, model2] = f"{model1_win_rate:.1f}%"
                    score_diff_data.loc[model1, model2] = f"{-score_diff:.2f}"
                    
                else:
                    # No results for this matchup
                    win_rate_data.loc[model1, model2] = "N/A"
                    score_diff_data.loc[model1, model2] = "N/A"
    
    # Create an Excel workbook
    wb = Workbook()
    
    # Create Summary sheet
    summary_sheet = wb.active
    summary_sheet.title = "Summary"
    
    # Add headers
    summary_sheet['A1'] = "Team 0 Agent Comparison Summary"
    summary_sheet['A1'].font = Font(bold=True, size=14)
    summary_sheet.merge_cells('A1:E1')
    
    summary_sheet['A3'] = "Model"
    summary_sheet['B3'] = "Episodes"
    summary_sheet['C3'] = "Average Win Rate (%)"
    summary_sheet['D3'] = "Average Score Diff"
    summary_sheet['E3'] = "Path"
    
    # Bold the headers
    for cell in summary_sheet['A3:E3'][0]:
        cell.font = Font(bold=True)
    
    # Fill summary data
    for row_idx, (name, episode, path) in enumerate(checkpoints_info, start=4):
        # Calculate average win rate and score diff against all other models
        win_rates = []
        score_diffs = []
        
        for other_name in model_names:
            if other_name != name:
                win_rate_str = win_rate_data.loc[name, other_name]
                score_diff_str = score_diff_data.loc[name, other_name]
                
                if win_rate_str != "N/A" and win_rate_str != "—":
                    win_rates.append(float(win_rate_str.strip('%')))
                
                if score_diff_str != "N/A" and score_diff_str != "—":
                    score_diffs.append(float(score_diff_str))
        
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else float('nan')
        avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else float('nan')
        
        summary_sheet[f'A{row_idx}'] = name
        summary_sheet[f'B{row_idx}'] = episode if episode != float('inf') else "Unknown"
        summary_sheet[f'C{row_idx}'] = f"{avg_win_rate:.1f}%" if not pd.isna(avg_win_rate) else "N/A"
        summary_sheet[f'D{row_idx}'] = f"{avg_score_diff:.2f}" if not pd.isna(avg_score_diff) else "N/A"
        summary_sheet[f'E{row_idx}'] = path
    
    # Auto-adjust column widths
    for column in summary_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = max_length + 2
        summary_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Create Win Rate sheet
    win_rate_sheet = wb.create_sheet(title="Win Rates")
    
    # Add title
    win_rate_sheet['A1'] = "Win Rates (Row vs Column)"
    win_rate_sheet['A1'].font = Font(bold=True, size=14)
    win_rate_sheet.merge_cells(f'A1:{chr(65 + len(model_names) + 1)}1')
    
    # Add a header for interpretation
    win_rate_sheet['A2'] = "Higher percentage means the row model wins more often against the column model"
    win_rate_sheet.merge_cells(f'A2:{chr(65 + len(model_names) + 1)}2')
    
    # Transform DataFrame to rows and add to sheet
    for r_idx, row in enumerate(dataframe_to_rows(win_rate_data, index=True, header=True), start=4):
        for c_idx, value in enumerate(row, start=1):
            win_rate_sheet.cell(row=r_idx, column=c_idx, value=value)
            
            # Apply color gradient to win rates (green for high, red for low)
            cell = win_rate_sheet.cell(row=r_idx, column=c_idx)
            if isinstance(value, str) and '%' in value and value != "—":
                try:
                    win_rate = float(value.strip('%'))
                    # Apply gradient: green (good) to red (bad)
                    if win_rate >= 50:
                        intensity = min(255, int(155 + (win_rate - 50) * 2))
                        green_hex = format(intensity, '02x')
                        red_hex = format(255 - intensity // 3, '02x')
                        cell.fill = PatternFill(start_color=f"{red_hex}{green_hex}55", end_color=f"{red_hex}{green_hex}55", fill_type="solid")
                    else:
                        intensity = min(255, int(155 + (50 - win_rate) * 2))
                        red_hex = format(intensity, '02x')
                        green_hex = format(255 - intensity // 3, '02x')
                        cell.fill = PatternFill(start_color=f"{red_hex}{green_hex}55", end_color=f"{red_hex}{green_hex}55", fill_type="solid")
                except ValueError:
                    pass
    
    # Apply formatting to header row and column
    for row in win_rate_sheet.iter_rows(min_row=4, max_row=4, min_col=2):
        for cell in row:
            cell.font = Font(bold=True)
    
    for col in win_rate_sheet.iter_cols(min_col=1, max_col=1, min_row=5):
        for cell in col:
            cell.font = Font(bold=True)
    
    # Auto-adjust column widths
    for column in win_rate_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = max_length + 2
        win_rate_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Create Score Difference sheet
    score_diff_sheet = wb.create_sheet(title="Score Differences")
    
    # Add title
    score_diff_sheet['A1'] = "Score Differences (Row minus Column)"
    score_diff_sheet['A1'].font = Font(bold=True, size=14)
    score_diff_sheet.merge_cells(f'A1:{chr(65 + len(model_names) + 1)}1')
    
    # Add a header for interpretation
    score_diff_sheet['A2'] = "Positive values mean the row model scores higher than the column model"
    score_diff_sheet.merge_cells(f'A2:{chr(65 + len(model_names) + 1)}2')
    
    # Transform DataFrame to rows and add to sheet
    for r_idx, row in enumerate(dataframe_to_rows(score_diff_data, index=True, header=True), start=4):
        for c_idx, value in enumerate(row, start=1):
            score_diff_sheet.cell(row=r_idx, column=c_idx, value=value)
            
            # Apply color gradient to score differences
            cell = score_diff_sheet.cell(row=r_idx, column=c_idx)
            if isinstance(value, str) and value not in ["N/A", "—"]:
                try:
                    score_diff = float(value)
                    # Apply gradient: green (positive) to red (negative)
                    if score_diff > 0:
                        intensity = min(255, int(155 + min(score_diff * 25, 100)))
                        green_hex = format(intensity, '02x')
                        red_hex = format(255 - intensity // 3, '02x')
                        cell.fill = PatternFill(start_color=f"{red_hex}{green_hex}55", end_color=f"{red_hex}{green_hex}55", fill_type="solid")
                    elif score_diff < 0:
                        intensity = min(255, int(155 + min(abs(score_diff) * 25, 100)))
                        red_hex = format(intensity, '02x')
                        green_hex = format(255 - intensity // 3, '02x')
                        cell.fill = PatternFill(start_color=f"{red_hex}{green_hex}55", end_color=f"{red_hex}{green_hex}55", fill_type="solid")
                except ValueError:
                    pass
    
    # Apply formatting to header row and column
    for row in score_diff_sheet.iter_rows(min_row=4, max_row=4, min_col=2):
        for cell in row:
            cell.font = Font(bold=True)
    
    for col in score_diff_sheet.iter_cols(min_col=1, max_col=1, min_row=5):
        for cell in col:
            cell.font = Font(bold=True)
    
    # Auto-adjust column widths
    for column in score_diff_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = max_length + 2
        score_diff_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Create a Detailed Results sheet
    detailed_sheet = wb.create_sheet(title="Detailed Results")
    
    # Add headers
    detailed_sheet['A1'] = "Detailed Matchup Results"
    detailed_sheet['A1'].font = Font(bold=True, size=14)
    detailed_sheet.merge_cells('A1:H1')
    
    row_idx = 3
    detailed_sheet[f'A{row_idx}'] = "Model A"
    detailed_sheet[f'B{row_idx}'] = "Model B"
    detailed_sheet[f'C{row_idx}'] = "A Wins"
    detailed_sheet[f'D{row_idx}'] = "B Wins"
    detailed_sheet[f'E{row_idx}'] = "Draws"
    detailed_sheet[f'F{row_idx}'] = "A Avg Score"
    detailed_sheet[f'G{row_idx}'] = "B Avg Score"
    detailed_sheet[f'H{row_idx}'] = "Score Diff (A-B)"
    
    # Bold the headers
    for cell in detailed_sheet[f'A{row_idx}:H{row_idx}'][0]:
        cell.font = Font(bold=True)
    
    # Add detailed results
    row_idx = 4
    for matchup, data in results.items():
        agents = matchup.split("_vs_")
        agent1_name, agent2_name = agents
        
        detailed_sheet[f'A{row_idx}'] = agent1_name
        detailed_sheet[f'B{row_idx}'] = agent2_name
        detailed_sheet[f'C{row_idx}'] = data['agent1_wins']
        detailed_sheet[f'D{row_idx}'] = data['agent2_wins']
        detailed_sheet[f'E{row_idx}'] = data['draws']
        detailed_sheet[f'F{row_idx}'] = data['agent1_avg_score']
        detailed_sheet[f'G{row_idx}'] = data['agent2_avg_score']
        detailed_sheet[f'H{row_idx}'] = data['agent1_avg_score'] - data['agent2_avg_score']
        
        row_idx += 1
    
    # Auto-adjust column widths
    for column in detailed_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = max_length + 2
        detailed_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Save the workbook
    wb.save(output_file)
    print(f"\nExcel comparison report saved to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate Team 0 Scopone AI checkpoints")
    parser.add_argument("--checkpoints", nargs="+", help="Paths to Team 0 checkpoint files or directories")
    parser.add_argument("--checkpoint_dir", help="Directory containing Team 0 checkpoints")
    parser.add_argument("--checkpoint_pattern", default="*team0*ep*.pth", help="Pattern to match Team 0 checkpoint files")
    parser.add_argument("--games", type=int, default=10000, help="Number of games to play for each matchup")
    parser.add_argument("--output", help="Output file path (default: auto-generated)")
    parser.add_argument("--excel", help="Excel output file path (default: auto-generated)")
    parser.add_argument("--limit", type=int, help="Limit the number of checkpoints to evaluate")
    args = parser.parse_args()
    
    # Get checkpoint paths
    checkpoint_paths = []
    
    if args.checkpoints:
        for cp in args.checkpoints:
            if os.path.isfile(cp):
                checkpoint_paths.append(cp)
            elif os.path.isdir(cp):
                checkpoint_paths.extend(find_checkpoints(cp, args.checkpoint_pattern))
            else:
                print(f"Warning: Checkpoint not found: {cp}")
    
    if args.checkpoint_dir:
        checkpoint_paths.extend(find_checkpoints(args.checkpoint_dir, args.checkpoint_pattern))
    
    # Remove duplicates
    checkpoint_paths = list(set(checkpoint_paths))
    
    # Ensure we only have Team 0 checkpoints
    team0_checkpoints = []
    for cp in checkpoint_paths:
        if "team0" in os.path.basename(cp):
            team0_checkpoints.append(cp)
        else:
            print(f"Skipping non-Team 0 checkpoint: {cp}")
    
    checkpoint_paths = team0_checkpoints
    
    # Limit number of checkpoints if requested
    if args.limit and len(checkpoint_paths) > args.limit:
        # Try to pick evenly spaced checkpoints
        step = len(checkpoint_paths) // args.limit
        limited_paths = [checkpoint_paths[i] for i in range(0, len(checkpoint_paths), step)][:args.limit]
        checkpoint_paths = limited_paths
    
    if not checkpoint_paths:
        print("Error: No Team 0 checkpoint files found. Please provide valid checkpoint paths.")
        return
    
    # Report found checkpoints
    print(f"Found {len(checkpoint_paths)} Team 0 checkpoint files:")
    for i, cp in enumerate(checkpoint_paths):
        print(f"{i+1}. {cp}")
    
    # Configure GPU if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = args.output if args.output else f"team0_comparison_{timestamp}.txt"
    excel_file = args.excel if args.excel else f"team0_comparison_{timestamp}.xlsx"
    
    with open(output_file, "w") as f:
        f.write(f"Scopone AI Team 0 Checkpoint Comparison Results ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Number of games per evaluation: {args.games}\n\n")
        f.write(f"Checkpoints evaluated ({len(checkpoint_paths)}):\n")
        for i, cp in enumerate(checkpoint_paths):
            episode = extract_episode_number(cp)
            episode_str = f" (Episode {episode})" if episode != float('inf') else ""
            f.write(f"{i+1}. {cp}{episode_str}\n")
        f.write("\n")
        
        # Only do comparison - no self-play needed since all agents are Team 0
        if len(checkpoint_paths) > 1:
            f.write("=== COMPARISON RESULTS ===\n\n")
            comparison_results = evaluate_checkpoints(checkpoint_paths, args.games)
            
            for matchup, data in comparison_results.items():
                agents = matchup.split("_vs_")
                
                # Calculate first-mover advantage
                first_starter_advantage = 0
                total_decided_games = data['agent1_wins'] + data['agent2_wins']
                if total_decided_games > 0:
                    first_starter_advantage = data['first_starter_wins'] / total_decided_games
                
                f.write(f"{agents[0]} vs {agents[1]}:\n")
                f.write(f"  {agents[0]} wins: {data['agent1_wins']} ({data['agent1_wins']/args.games*100:.1f}%)\n")
                f.write(f"  {agents[1]} wins: {data['agent2_wins']} ({data['agent2_wins']/args.games*100:.1f}%)\n")
                f.write(f"  First-starter advantage: {first_starter_advantage:.2f} (1.0 = 100% advantage, 0.5 = no advantage)\n")
                f.write(f"  Draws: {data['draws']} ({data['draws']/args.games*100:.1f}%)\n")
                f.write(f"  Average score {agents[0]}: {data['agent1_avg_score']:.2f}\n")
                f.write(f"  Average score {agents[1]}: {data['agent2_avg_score']:.2f}\n")
                f.write(f"  Average game length: {sum(data['game_lengths'])/len(data['game_lengths']):.1f} moves\n")
                f.write(f"  Score distributions:\n")
                f.write(f"    {agents[0]}: {dict(data['agent1_score_distribution'])}\n")
                f.write(f"    {agents[1]}: {dict(data['agent2_score_distribution'])}\n\n")
            
            # Generate Excel report with comparative matrix
            generate_excel_comparison(checkpoint_paths, comparison_results, excel_file)
    
    print(f"\nResults saved to {output_file}")
    print(f"Excel comparison report saved to {excel_file}")

if __name__ == "__main__":
    main()