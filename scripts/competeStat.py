import os
import json
import glob
from typing import Dict, List, Tuple

def load_competition_results(results_dir: str = "competeRes") -> Dict[str, List[Dict]]:
    """Load all competition results from JSON files, categorized by ActorParamClamped condition."""
    results = {"clamped": [], "unclamped": []}
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                
                # Determine condition from filename
                filename = os.path.basename(file_path)
                if "ActorParamClamped__True" in filename:
                    results["clamped"].append(result)
                elif "ActorParamClamped__False" in filename:
                    results["unclamped"].append(result)
                else:
                    print(f"Warning: Could not determine condition for {filename}")
                    
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def analyze_winning_rates(results: List[Dict]) -> Dict:
    """Analyze winning rates from competition results."""
    if not results:
        print("No results found!")
        return {}
    
    # Count wins for each agent
    first_0_wins = 0  # Separable agent
    second_0_wins = 0  # Entangled agent
    ties = 0
    total_games = len(results)
    
    # Detailed statistics
    first_0_scores = []
    second_0_scores = []
    
    for result in results:
        winner = result.get("winner")
        final_return = result.get("final_return", {})
        
        first_0_score = final_return.get("first_0", 0)
        second_0_score = final_return.get("second_0", 0)
        
        first_0_scores.append(first_0_score)
        second_0_scores.append(second_0_score)
        
        if winner == "first_0":
            first_0_wins += 1
        elif winner == "second_0":
            second_0_wins += 1
        else:
            ties += 1
    
    # Calculate statistics
    stats = {
        "total_games": total_games,
        "separable_agent_wins": first_0_wins,
        "entangled_agent_wins": second_0_wins,
        "ties": ties,
        "separable_win_rate": first_0_wins / total_games if total_games > 0 else 0,
        "entangled_win_rate": second_0_wins / total_games if total_games > 0 else 0,
        "tie_rate": ties / total_games if total_games > 0 else 0,
        "avg_separable_score": sum(first_0_scores) / len(first_0_scores) if first_0_scores else 0,
        "avg_entangled_score": sum(second_0_scores) / len(second_0_scores) if second_0_scores else 0,
        "max_separable_score": max(first_0_scores) if first_0_scores else 0,
        "max_entangled_score": max(second_0_scores) if second_0_scores else 0,
        "min_separable_score": min(first_0_scores) if first_0_scores else 0,
        "min_entangled_score": min(second_0_scores) if second_0_scores else 0
    }
    
    return stats

def print_statistics(stats: Dict, condition: str):
    """Print formatted statistics for a specific condition."""
    if not stats:
        print(f"No statistics to display for {condition}.")
        return
    
    condition_title = "CLAMPED (ActorParamClamped=True)" if condition == "clamped" else "UNCLAMPED (ActorParamClamped=False)"
    
    print("=" * 70)
    print(f"COMPETITION STATISTICS: {condition_title}")
    print("=" * 70)
    print(f"Total Games Played: {stats['total_games']}")
    print()
    
    print("WINNING RATES:")
    print(f"  Separable Agent (first_0): {stats['separable_agent_wins']}/{stats['total_games']} ({stats['separable_win_rate']:.2%})")
    print(f"  Entangled Agent (second_0): {stats['entangled_agent_wins']}/{stats['total_games']} ({stats['entangled_win_rate']:.2%})")
    print(f"  Ties: {stats['ties']}/{stats['total_games']} ({stats['tie_rate']:.2%})")
    print()
    
    print("SCORE STATISTICS:")
    print(f"  Average Scores:")
    print(f"    Separable Agent: {stats['avg_separable_score']:.2f}")
    print(f"    Entangled Agent: {stats['avg_entangled_score']:.2f}")
    print()
    
    print(f"  Score Ranges:")
    print(f"    Separable Agent: {stats['min_separable_score']} to {stats['max_separable_score']}")
    print(f"    Entangled Agent: {stats['min_entangled_score']} to {stats['max_entangled_score']}")
    print()
    
    # Determine overall performance
    if stats['entangled_win_rate'] > stats['separable_win_rate']:
        better_agent = "Entangled Agent"
        advantage = stats['entangled_win_rate'] - stats['separable_win_rate']
    elif stats['separable_win_rate'] > stats['entangled_win_rate']:
        better_agent = "Separable Agent"
        advantage = stats['separable_win_rate'] - stats['entangled_win_rate']
    else:
        better_agent = "Neither (tied)"
        advantage = 0
    
    print("SUMMARY:")
    if advantage > 0:
        print(f"  {better_agent} performs better with a {advantage:.2%} advantage")
    else:
        print(f"  Both agents perform equally well")
    print("=" * 70)
    print()

def print_comparison(clamped_stats: Dict, unclamped_stats: Dict):
    """Print comparison between clamped and unclamped conditions."""
    print("=" * 70)
    print("COMPARISON: CLAMPED vs UNCLAMPED CONDITIONS")
    print("=" * 70)
    
    if not clamped_stats or not unclamped_stats:
        print("Cannot compare - missing data for one or both conditions")
        return
    
    print("ENTANGLED AGENT PERFORMANCE:")
    print(f"  Clamped:   {clamped_stats['entangled_win_rate']:.2%} win rate")
    print(f"  Unclamped: {unclamped_stats['entangled_win_rate']:.2%} win rate")
    
    entangled_improvement = unclamped_stats['entangled_win_rate'] - clamped_stats['entangled_win_rate']
    if entangled_improvement > 0:
        print(f"  → Entangled agent performs {entangled_improvement:.2%} better when unclamped")
    elif entangled_improvement < 0:
        print(f"  → Entangled agent performs {abs(entangled_improvement):.2%} better when clamped")
    else:
        print(f"  → No difference in entangled agent performance")
    print()
    
    print("SEPARABLE AGENT PERFORMANCE:")
    print(f"  Clamped:   {clamped_stats['separable_win_rate']:.2%} win rate")
    print(f"  Unclamped: {unclamped_stats['separable_win_rate']:.2%} win rate")
    
    separable_improvement = unclamped_stats['separable_win_rate'] - clamped_stats['separable_win_rate']
    if separable_improvement > 0:
        print(f"  → Separable agent performs {separable_improvement:.2%} better when unclamped")
    elif separable_improvement < 0:
        print(f"  → Separable agent performs {abs(separable_improvement):.2%} better when clamped")
    else:
        print(f"  → No difference in separable agent performance")
    print()
    
    print("OVERALL IMPACT OF CLAMPING:")
    clamped_advantage = clamped_stats['entangled_win_rate'] - clamped_stats['separable_win_rate']
    unclamped_advantage = unclamped_stats['entangled_win_rate'] - unclamped_stats['separable_win_rate']
    
    print(f"  Clamped:   Entangled advantage = {clamped_advantage:.2%}")
    print(f"  Unclamped: Entangled advantage = {unclamped_advantage:.2%}")
    
    if unclamped_advantage > clamped_advantage:
        print(f"  → Unclamping favors the entangled agent by {unclamped_advantage - clamped_advantage:.2%}")
    elif clamped_advantage > unclamped_advantage:
        print(f"  → Clamping favors the entangled agent by {clamped_advantage - unclamped_advantage:.2%}")
    else:
        print(f"  → Clamping has no net effect on relative performance")
    
    print("=" * 70)

def save_markdown_report(clamped_stats: Dict, unclamped_stats: Dict, output_file: str = "competition_report.md"):
    """Save a detailed markdown report of the competition statistics."""
    with open(output_file, 'w') as f:
        f.write("# Competition Statistics Report\n\n")
        f.write("This report analyzes the performance of Entangled vs Separable agents in Pong competition.\n\n")
        
        # Clamped condition section
        if clamped_stats:
            f.write("## Clamped Condition (ActorParamClamped=True)\n\n")
            f.write(f"**Total Games Played:** {clamped_stats['total_games']}\n\n")
            
            f.write("### Winning Rates\n\n")
            f.write("| Agent | Wins | Win Rate |\n")
            f.write("|-------|------|----------|\n")
            f.write(f"| Separable Agent (first_0) | {clamped_stats['separable_agent_wins']}/{clamped_stats['total_games']} | {clamped_stats['separable_win_rate']:.2%} |\n")
            f.write(f"| Entangled Agent (second_0) | {clamped_stats['entangled_agent_wins']}/{clamped_stats['total_games']} | {clamped_stats['entangled_win_rate']:.2%} |\n")
            f.write(f"| Ties | {clamped_stats['ties']}/{clamped_stats['total_games']} | {clamped_stats['tie_rate']:.2%} |\n\n")
            
            f.write("### Score Statistics\n\n")
            f.write("| Metric | Separable Agent | Entangled Agent |\n")
            f.write("|--------|-----------------|------------------|\n")
            f.write(f"| Average Score | {clamped_stats['avg_separable_score']:.2f} | {clamped_stats['avg_entangled_score']:.2f} |\n")
            f.write(f"| Min Score | {clamped_stats['min_separable_score']} | {clamped_stats['min_entangled_score']} |\n")
            f.write(f"| Max Score | {clamped_stats['max_separable_score']} | {clamped_stats['max_entangled_score']} |\n\n")
            
            # Clamped summary
            if clamped_stats['entangled_win_rate'] > clamped_stats['separable_win_rate']:
                better_agent = "Entangled Agent"
                advantage = clamped_stats['entangled_win_rate'] - clamped_stats['separable_win_rate']
            elif clamped_stats['separable_win_rate'] > clamped_stats['entangled_win_rate']:
                better_agent = "Separable Agent"
                advantage = clamped_stats['separable_win_rate'] - clamped_stats['entangled_win_rate']
            else:
                better_agent = "Neither (tied)"
                advantage = 0
            
            f.write("### Summary\n\n")
            if advantage > 0:
                f.write(f"**{better_agent}** performs better with a **{advantage:.2%}** advantage.\n\n")
            else:
                f.write("Both agents perform equally well.\n\n")
        
        # Unclamped condition section
        if unclamped_stats:
            f.write("## Unclamped Condition (ActorParamClamped=False)\n\n")
            f.write(f"**Total Games Played:** {unclamped_stats['total_games']}\n\n")
            
            f.write("### Winning Rates\n\n")
            f.write("| Agent | Wins | Win Rate |\n")
            f.write("|-------|------|----------|\n")
            f.write(f"| Separable Agent (first_0) | {unclamped_stats['separable_agent_wins']}/{unclamped_stats['total_games']} | {unclamped_stats['separable_win_rate']:.2%} |\n")
            f.write(f"| Entangled Agent (second_0) | {unclamped_stats['entangled_agent_wins']}/{unclamped_stats['total_games']} | {unclamped_stats['entangled_win_rate']:.2%} |\n")
            f.write(f"| Ties | {unclamped_stats['ties']}/{unclamped_stats['total_games']} | {unclamped_stats['tie_rate']:.2%} |\n\n")
            
            f.write("### Score Statistics\n\n")
            f.write("| Metric | Separable Agent | Entangled Agent |\n")
            f.write("|--------|-----------------|------------------|\n")
            f.write(f"| Average Score | {unclamped_stats['avg_separable_score']:.2f} | {unclamped_stats['avg_entangled_score']:.2f} |\n")
            f.write(f"| Min Score | {unclamped_stats['min_separable_score']} | {unclamped_stats['min_entangled_score']} |\n")
            f.write(f"| Max Score | {unclamped_stats['max_separable_score']} | {unclamped_stats['max_entangled_score']} |\n\n")
            
            # Unclamped summary
            if unclamped_stats['entangled_win_rate'] > unclamped_stats['separable_win_rate']:
                better_agent = "Entangled Agent"
                advantage = unclamped_stats['entangled_win_rate'] - unclamped_stats['separable_win_rate']
            elif unclamped_stats['separable_win_rate'] > unclamped_stats['entangled_win_rate']:
                better_agent = "Separable Agent"
                advantage = unclamped_stats['separable_win_rate'] - unclamped_stats['entangled_win_rate']
            else:
                better_agent = "Neither (tied)"
                advantage = 0
            
            f.write("### Summary\n\n")
            if advantage > 0:
                f.write(f"**{better_agent}** performs better with a **{advantage:.2%}** advantage.\n\n")
            else:
                f.write("Both agents perform equally well.\n\n")
        
        # Comparison section
        if clamped_stats and unclamped_stats:
            f.write("## Comparison: Clamped vs Unclamped Conditions\n\n")
            
            f.write("### Agent Performance Comparison\n\n")
            f.write("| Agent | Clamped Win Rate | Unclamped Win Rate | Improvement |\n")
            f.write("|-------|------------------|--------------------|--------------|\n")
            
            entangled_improvement = unclamped_stats['entangled_win_rate'] - clamped_stats['entangled_win_rate']
            separable_improvement = unclamped_stats['separable_win_rate'] - clamped_stats['separable_win_rate']
            
            f.write(f"| Entangled Agent | {clamped_stats['entangled_win_rate']:.2%} | {unclamped_stats['entangled_win_rate']:.2%} | {entangled_improvement:+.2%} |\n")
            f.write(f"| Separable Agent | {clamped_stats['separable_win_rate']:.2%} | {unclamped_stats['separable_win_rate']:.2%} | {separable_improvement:+.2%} |\n\n")
            
            f.write("### Performance Analysis\n\n")
            
            # Entangled agent analysis
            if entangled_improvement > 0:
                f.write(f"- **Entangled agent** performs **{entangled_improvement:.2%} better** when unclamped\n")
            elif entangled_improvement < 0:
                f.write(f"- **Entangled agent** performs **{abs(entangled_improvement):.2%} better** when clamped\n")
            else:
                f.write("- **Entangled agent** shows no difference between conditions\n")
            
            # Separable agent analysis
            if separable_improvement > 0:
                f.write(f"- **Separable agent** performs **{separable_improvement:.2%} better** when unclamped\n")
            elif separable_improvement < 0:
                f.write(f"- **Separable agent** performs **{abs(separable_improvement):.2%} better** when clamped\n")
            else:
                f.write("- **Separable agent** shows no difference between conditions\n")
            
            f.write("\n### Overall Impact of Clamping\n\n")
            clamped_advantage = clamped_stats['entangled_win_rate'] - clamped_stats['separable_win_rate']
            unclamped_advantage = unclamped_stats['entangled_win_rate'] - unclamped_stats['separable_win_rate']
            
            f.write("| Condition | Entangled Advantage |\n")
            f.write("|-----------|---------------------|\n")
            f.write(f"| Clamped | {clamped_advantage:+.2%} |\n")
            f.write(f"| Unclamped | {unclamped_advantage:+.2%} |\n\n")
            
            if unclamped_advantage > clamped_advantage:
                f.write(f"🔍 **Key Finding:** Unclamping favors the entangled agent by **{unclamped_advantage - clamped_advantage:.2%}**\n\n")
            elif clamped_advantage > unclamped_advantage:
                f.write(f"🔍 **Key Finding:** Clamping favors the entangled agent by **{clamped_advantage - unclamped_advantage:.2%}**\n\n")
            else:
                f.write("🔍 **Key Finding:** Clamping has no net effect on relative performance\n\n")
        
        # Footer
        f.write("---\n")
        f.write("*Report generated automatically by competeStat.py*\n")
    
    print(f"Markdown report saved to {output_file}")

def save_statistics(clamped_stats: Dict, unclamped_stats: Dict, output_file: str = "competition_statistics.json"):
    """Save statistics to a JSON file."""
    combined_stats = {
        "clamped_condition": clamped_stats,
        "unclamped_condition": unclamped_stats,
        "comparison": {
            "entangled_improvement_unclamped": unclamped_stats.get('entangled_win_rate', 0) - clamped_stats.get('entangled_win_rate', 0) if clamped_stats and unclamped_stats else 0,
            "separable_improvement_unclamped": unclamped_stats.get('separable_win_rate', 0) - clamped_stats.get('separable_win_rate', 0) if clamped_stats and unclamped_stats else 0
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(combined_stats, f, indent=4)
    print(f"Statistics saved to {output_file}")

def main():
    """Main function to run the competition analysis."""
    # Load results from the competeRes directory
    results_dir = "competeRes"
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    print(f"Loading competition results from {results_dir}...")
    categorized_results = load_competition_results(results_dir)
    
    clamped_results = categorized_results["clamped"]
    unclamped_results = categorized_results["unclamped"]
    
    print(f"Found {len(clamped_results)} clamped results")
    print(f"Found {len(unclamped_results)} unclamped results")
    print()
    
    # Analyze each condition
    clamped_stats = analyze_winning_rates(clamped_results) if clamped_results else {}
    unclamped_stats = analyze_winning_rates(unclamped_results) if unclamped_results else {}
    
    # Print statistics for each condition
    if clamped_stats:
        print_statistics(clamped_stats, "clamped")
    
    if unclamped_stats:
        print_statistics(unclamped_stats, "unclamped")
    
    # Print comparison if both conditions have data
    if clamped_stats and unclamped_stats:
        print_comparison(clamped_stats, unclamped_stats)
    
    # Save statistics (JSON and Markdown)
    save_statistics(clamped_stats, unclamped_stats)
    save_markdown_report(clamped_stats, unclamped_stats)

if __name__ == "__main__":
    main()