import sys

def get_freq_dist(input_file):
    d = {i: [0] * 9 for i in range(10)}
    cnt = 0
    with open(input_file, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split(",")[1].strip()
            line = [int(i) for i in line.split(" ")]
            for idx, count in enumerate(line):
                d[idx][count] += 1
    d = {i: [j / sum(value) for j in value] for i, value in d.items()}
    for i in range(10):
        print(f"{i}: {d[i]}")

if __name__=="__main__":
    get_freq_dist(sys.argv[1])