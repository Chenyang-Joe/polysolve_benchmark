fname = "/u/1/chenyang/benchmark_data/exp-result/20250628_logging_expand/AMGCL/2-cubes.log"
# fname = "/u/1/chenyang/benchmark_data/exp-result/20250628_logging_expand/Hypre/5-cubes.log"
# fname = "/u/1/chenyang/benchmark_data/exp-result/20250628_logging_expand/Hypre/circle-mat.log"
count = 0
begin_line = []
end_line = []
with open(fname, 'r') as f:
    lines = f.readlines()  # read all lines into a list

for i, line in enumerate(lines):
    if "[EXPBEGIN]" in line:
        begin_line.append(i)
    if "[EXPEND]" in line:
        end_line.append(i)

len_begin = len(begin_line)
len_end = len(end_line)
if len_begin != len_end:
    print("#[EXPBEGIN] #[EXPEND] mismatches, unknown error formats.")

begin_correct = []
end_correct = []

for begin_i, end_i in zip(begin_line, end_line):
    error_detected = False
    for j in range(begin_i, end_i):
        if "ERROR" in lines[j] or "error" in lines[j]:
            print(f"errors between {begin_i+1} line and {end_i+1} line")
            error_detected = True
            break
    if error_detected:
        continue 
    begin_correct.append(begin_i)
    end_correct.append(end_i)

    seq = []
    for begin_i, end_i in zip(begin_correct, end_correct):
        element = {}
        element["solver_tol"] = lines[begin_i+6].split(" ")[-1]
        print(element["solver_tol"])




