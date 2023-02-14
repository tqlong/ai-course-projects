import sys
import re

if __name__ == "__main__":
    score, total = 0, 0
    for line in sys.stdin:
        x = re.search("^Question q[0-9]+: ([0-9]+)/([0-9]+)", line)
        if x is not None:
            if len(x.groups()) == 2:
                score += int(x.groups()[0])
                total += int(x.groups()[1])
    
    print(f'score = {score} total = {total} {"FAIL" if score < total else "SUCCESS"}')