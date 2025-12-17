import os
import numpy as np

def sequence_data(fname, solver):
    begin_line = []
    end_line = []
    with open(fname, 'r') as f:
        lines = f.readlines()  # read all lines into a list

    for i, line in enumerate(lines):
        if "[EXPBEGIN]"  in line:
            begin_line.append(i)
        if "[EXPEND]" in line:
            end_line.append(i)
        if "(core dumped)" in line:
            end_line.append(i)
            # print(f"Aborted detected in log file {fname} at line {i+1}, marking experiment end.")

    len_begin = len(begin_line)
    len_end = len(end_line)
    assert len_begin == len_end, f"#[EXPBEGIN] #[EXPEND] mismatches, unknown error formats. Error dir: {fname}"

    begin_correct = []
    end_correct = []

    error_detected_log = False
    for begin_i, end_i in zip(begin_line, end_line):
        error_detected_exp = False

        for j in range(begin_i, end_i):
            if "ERROR" in lines[j] or "error" in lines[j] or "Error" in lines[j] or "TIMEOUT" in lines[j]:
#                 print(f"errors between {begin_i+1} line and {end_i+1} line")
                error_detected_exp = True
                break

        if error_detected_exp:
            error_detected_log = True
            continue 

        begin_correct.append(begin_i)
        end_correct.append(end_i)
        
    if error_detected_log:
        print(f"Warning: errors detected in log file {fname}, some experiments are skipped.")
        # raise ValueError(f"error in current exp: {fname}")


    seq = []
    for begin_i, end_i in zip(begin_correct, end_correct):
        element = {"solver": solver}
        cmd = lines[begin_i-3].split(" ")
        for part in cmd:
            if "_A.bin" in part:
                element["bin_A"] = part.strip()
            if "_b.bin" in part:
                element["bin_b"] = part.strip()

        if element.get("bin_A") is None or element.get("bin_b") is None:
            raise ValueError(f"cannot find bin_A or bin_b in log file {fname}")

        if solver == "Eigen::PardisoLDLT": # direct solver
            element["residual"] = float(lines[end_i-5].split(" ")[-1].strip())
            element["outer"] = float(lines[end_i-4].split(" ")[-1].strip())
            element["inner"] = float(lines[end_i-3].split(" ")[-1].strip())
            element["clock_time"] = float(lines[end_i-2].split(" ")[-1].strip().rstrip("s"))
            element["elapse_time"] = float(lines[end_i-1].split(" ")[-1].strip().rstrip("s"))

        elif solver == "Hypre" : # iterative solver
            element["residual"] = float(lines[end_i-10].split(" ")[-1].strip())
            element["outer"] = float(lines[end_i-9].split(" ")[-1].strip())
            element["inner"] = float(lines[end_i-8].split(" ")[-1].strip())
            element["solver_tol"] = float(lines[end_i-7].split(" ")[-1].strip())
            element["solver_maxiter"] = float(lines[end_i-6].split(" ")[-1].strip())
            element["final_res_norm"] = float(lines[end_i-5].split(" ")[-1].strip())
            element["num_iterations"] = float(lines[end_i-4].split(" ")[-1].strip())
            element["norm_b"] = float(lines[end_i-3].split(" ")[-1].strip())
            element["clock_time"] = float(lines[end_i-2].split(" ")[-1].strip().rstrip("s"))
            element["elapse_time"] = float(lines[end_i-1].split(" ")[-1].strip().rstrip("s"))
        elif solver == "AMGCL":
            element["factorize"] = float(lines[end_i-12].split(" ")[-1].strip().rstrip("s"))
            element["solve"] = float(lines[end_i-11].split(" ")[-1].strip().rstrip("s"))
            element["residual"] = float(lines[end_i-10].split(" ")[-1].strip())
            element["outer"] = float(lines[end_i-9].split(" ")[-1].strip())
            element["inner"] = float(lines[end_i-8].split(" ")[-1].strip())
            element["solver_tol"] = float(lines[end_i-7].split(" ")[-1].strip())
            element["solver_maxiter"] = float(lines[end_i-6].split(" ")[-1].strip())
            element["final_res_norm"] = float(lines[end_i-5].split(" ")[-1].strip())
            element["num_iterations"] = float(lines[end_i-4].split(" ")[-1].strip())
            element["norm_b"] = float(lines[end_i-3].split(" ")[-1].strip())
            element["clock_time"] = float(lines[end_i-2].split(" ")[-1].strip().rstrip("s"))
            element["elapse_time"] = float(lines[end_i-1].split(" ")[-1].strip().rstrip("s"))
        else:
            raise ValueError(f"solver {solver} not recognized.")
        seq.append(element)
    return seq


def get_mat_sz(fp):
    try:
        # Read header: dim, is_spd, is_sequence
        header = np.fromfile(fp, dtype=np.int32, count=3, offset=0)
        # dim = header[0]
        # is_spd = header[1]
        # is_sequence = header[2]
        
        # Check if format_marker exists (POLYSOLVE_LARGE_INDEX format)
        format_check = np.fromfile(fp, dtype=np.int32, count=1, offset=3*4)
        
        if len(format_check) > 0 and format_check[0] == -1:
            # New format with ptrdiff_t (8 bytes per value on 64-bit systems)
            # Read: rows, cols, nnz, innS, outS
            dims = np.fromfile(fp, dtype=np.int64, count=5, offset=4*4)
            n_rows = dims[0]
        else:
            # Legacy format with int32 (4 bytes per value)
            # format_check[0] is actually the rows value
            # Read: rows (already read), cols, nnz, innS, outS
            n_rows = format_check[0]
        
        return int(n_rows)
    except Exception as e:
        print(f"mat size exception: {e}")
        return None

def get_nnz(fp):
    try:
        # Read header: dim, is_spd, is_sequence
        header = np.fromfile(fp, dtype=np.int32, count=3, offset=0)
        
        # Check if format_marker exists (POLYSOLVE_LARGE_INDEX format)
        format_check = np.fromfile(fp, dtype=np.int32, count=1, offset=3*4)
        
        if len(format_check) > 0 and format_check[0] == -1:
            # New format with ptrdiff_t (8 bytes per value)
            # Read: rows, cols, nnz, innS, outS
            dims = np.fromfile(fp, dtype=np.int64, count=5, offset=4*4)
            nnz = dims[2]
        else:
            # Legacy format with int32 (4 bytes per value)
            # format_check[0] is rows, read cols and nnz
            remaining = np.fromfile(fp, dtype=np.int32, count=4, offset=3*4)
            # remaining = [rows, cols, nnz, innS, outS]
            nnz = remaining[2]
        
        return int(nnz)
    except Exception as e:
        print(f"nnz exception: {e}")
        return None

def get_density(nnz, mat_sz):
    if nnz is None or mat_sz is None:
        return None
    else:
        nnz =  np.float128(nnz)
        mat_sz =  np.float128(mat_sz)
        return (nnz + 1e-10) / (mat_sz * mat_sz + 1e-10)

def get_sparsity(nnz, mat_sz):
    if nnz is None or mat_sz is None:
        return None
    else:
        nnz =  np.float128(nnz)
        mat_sz =  np.float128(mat_sz)
        return (mat_sz * mat_sz - nnz) / (mat_sz * mat_sz)



def parse_log_file(fname, solver):
    seq = sequence_data(fname, solver)

    log_data = []
    for entry in seq:
        data_one_exp = {}
        data_one_exp["log_path"] = fname
        data_one_exp["solver"] = solver
        data_one_exp["bin_A"] = entry["bin_A"]
        data_one_exp["bin_b"] = entry["bin_b"]
        data_one_exp["mat_sz"] = get_mat_sz(entry["bin_A"])
        data_one_exp["nnz"] = get_nnz(entry["bin_A"])
        data_one_exp["density"] = get_density(data_one_exp["nnz"], data_one_exp["mat_sz"])
        data_one_exp["sparsity"] = get_sparsity(data_one_exp["nnz"], data_one_exp["mat_sz"])


        data_one_exp["outer"] = entry["outer"]
        data_one_exp["inner"] = entry["inner"]
        data_one_exp["residual"] = entry["residual"]
        data_one_exp["clock_time"] = entry["clock_time"]
        data_one_exp["elapse_time"] = entry["elapse_time"]

        if solver == "AMGCL":
            data_one_exp["factorize"] = entry["factorize"]
            data_one_exp["solve"] = entry["solve"]
            data_one_exp["solver_tol"] = entry["solver_tol"]
            data_one_exp["solver_maxiter"] = entry["solver_maxiter"]
            data_one_exp["final_res_norm"] = entry["final_res_norm"]
            data_one_exp["num_iterations"] = entry["num_iterations"]
        elif solver == "Hypre":
            data_one_exp["solver_tol"] = entry["solver_tol"]
            data_one_exp["solver_maxiter"] = entry["solver_maxiter"]
            data_one_exp["final_res_norm"] = entry["final_res_norm"]
            data_one_exp["num_iterations"] = entry["num_iterations"]
        elif solver == "Eigen::PardisoLDLT":
            pass
        else:
            raise ValueError(f"solver {solver} not recognized.")
        log_data.append(data_one_exp)


    return log_data