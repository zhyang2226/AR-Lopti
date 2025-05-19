import re

def compute_score(solution_str , ground_truth) -> float:
    
    solution_str = solution_str.rstrip("<|endoftext|>")

    if '<answer>' in solution_str:
            result = re.split(r'<answer>', solution_str)[1]
    else:
        result = solution_str[len(solution_str) - 30:]
    
    try:
        correct = str(int(ground_truth)) in result
    except:
        correct = ground_truth in result
        
    return float(correct)

