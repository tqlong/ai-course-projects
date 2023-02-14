import sys
if __name__ == "__main__":
    for line in sys.stdin:
        if "FAIL" in line:
            exit(1)
