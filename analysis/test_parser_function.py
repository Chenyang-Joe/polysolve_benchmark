from libs.parser import *
import matplotlib.pyplot as plt
from collections import defaultdict





if __name__ == "__main__":
    trial_dir = "/u/1/chenyang/benchmark_data/larger_matrix_exp/larger_mat_exp_result/trial_2"
    data_all = []   
    for fname in os.listdir(trial_dir):
        if "Pardiso" in fname:
            solver = "Eigen::PardisoLDLT"
        elif "AMGCL" in fname:
            solver = "AMGCL"
        elif "Hypre" in fname:
            solver = "Hypre"

        log_path = os.path.join(trial_dir, fname)
        data = parse_log_file(log_path, solver)
        data_all.extend(data)

    # for each element in data_all
    # plot wall-clock time vs mat_sz
    # color them by solver+tolerance
    # e.g. AMGCL-1e-8, Hypre-1e-8, AMGCL-1e-10, Hypre-1e-10, Pardiso
    
    # Group data by solver+tolerance
    grouped_data = defaultdict(lambda: {"mat_sz": [], "clock_time": []})
    
    for entry in data_all:
        solver = entry["solver"]
        mat_sz = entry["mat_sz"]
        clock_time = entry["clock_time"]
        
        # Create label based on solver and tolerance
        if solver == "Eigen::PardisoLDLT":
            label = "Pardiso"
        else:
            tol = entry.get("solver_tol", None)
            if tol is not None:
                label = f"{solver}-{tol:.0e}"
            else:
                label = solver
        
        grouped_data[label]["mat_sz"].append(mat_sz)
        grouped_data[label]["clock_time"].append(clock_time)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    for label, data in sorted(grouped_data.items()):
        plt.scatter(data["mat_sz"], data["clock_time"], label=label, alpha=0.7, s=50)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Matrix Size")
    plt.ylabel("Wall-clock Time (s)")
    plt.title("Wall-clock Time vs Matrix Size (Log-Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig("walltime_vs_matsize.png", dpi=300)
    plt.show()
    
    print(f"Total data points: {len(data_all)}")
    print(f"Solver+Tolerance combinations: {list(grouped_data.keys())}")