import re

def main():
    with open('round3_algo.py', 'r') as f:
        content = f.read()
    
    # 1. Add bs_delta
    bs_delta_code = """
def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)
"""
    if 'def bs_delta' not in content:
        content = content.replace('def bs_call', bs_delta_code.strip() + '\n\ndef bs_call')

    with open('round3_algo.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
