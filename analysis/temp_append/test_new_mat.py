print("hello")

import os
import re
import subprocess
import resource

def run_cmd(cmd_string, timeout=30*60):

    print(cmd_string)
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)
    format = 'utf-8'

    memory_usage_mb = 0.0
    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error : " + str(msg.decode(format))
        else:
            code = 0
            msg = str(msg.decode(format))
            
            # Get the memory usage of the subprocess
            max_memory_bytes = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            memory_usage_mb = max_memory_bytes / (1024 * 1024) 

    except subprocess.TimeoutExpired:
        # 注意：不能使用p.kill和p.terminate，无法杀干净所有的子进程，需要使用os.killpg
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGUSR1)
 
        # 注意：如果开启下面这两行的话，会等到执行完成才报超时错误，但是可以输出执行结果
        # (outs, errs) = p.communicate()
        # print(outs.decode('utf-8'))
 
        code = 1
        msg = "[TIMEOUT] after " + str(round(timeout/60)) + " min"

    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)
 
    # print(msg)
    return code, msg, memory_usage_mb

# solver_list = ["Hypre", "AMGCL", "Eigen::PardisoLDLT"]
# solver_list = ["Eigen::PardisoLDLT"]
solver_list = ["Hypre"]
mat_dir = "/u/1/chenyang/benchmark_data/larger_matrix_exp/new_mat_bin/3D_golf_ball_trial3"
polysolve_bin = "/u/1/chenyang/benchmark/build/TestMatLogger"
# polysolve_bin = "/u/1/chenyang/benchmark/build/TestMatTime"
log_save_dir = "/u/1/chenyang/benchmark/analysis/temp_append/data2"
timeout = 30

pattern = re.compile(r'^\d+_\d+_A\.bin$')
bin_list = [f for f in os.listdir(mat_dir) if os.path.isfile(os.path.join(mat_dir, f)) and pattern.match(f)]

for solver in solver_list:
    log_path = os.path.join(log_save_dir, solver+".log")
    open(log_path, 'w').close() 


os.environ['OMP_NUM_THREADS'] = "1"

for f in bin_list:
    A = os.path.join(mat_dir, f)
    b = os.path.join(mat_dir, f.split(".")[0][:-1]+"b.bin")

    for solver in solver_list:
        cmd_string="%s %s %s %s"%(
            polysolve_bin,
            A,
            b,
            solver)  

        code,msg,mem=run_cmd(cmd_string, timeout*60)  # timeout=30*60sec

        log_path = os.path.join(log_save_dir, solver+".log")
        with open(log_path, 'a') as f:
            f.write(msg)


