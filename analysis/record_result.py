import subprocess
import os
import signal
import pandas as pd



def run_cmd(cmd_string, timeout=30*60):

    print(cmd_string)
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)
    format = 'utf-8'

    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error : " + str(msg.decode(format))
        else:
            code = 0
            msg = str(msg.decode(format))
            
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
    return code, msg, 

test = "/u/1/chenyang/benchmark/build/TestMatTime" # this is a excutable program written by c++
data_dir = "/u/1/chenyang/benchmark/analysis/data/time_vs_thread"

# AMGCL time vs # threads
# A = "/u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/74_2_A.bin"
# b = "/u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/74_2_b.bin"
A = "/u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/75_4_A.bin"
b = "/u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/75_4_b.bin"
# solver = "Eigen::PardisoLU"
solver = "Hypre"

cmd_string="%s %s %s %s"%(
                                    test,
                                    A,
                                    b,
                                    solver)   

threads = range(1,100,5)
# threads = [91]
results = []
for trial in range(5):
    for thread in threads:
        os.environ['OMP_NUM_THREADS'] = str(thread)
        print("Current threads: ", os.environ['OMP_NUM_THREADS'])

        code, msg = run_cmd(cmd_string, timeout=30*60)
        msg_split = msg.split(" ")
        clock_time = msg_split[1]
        wall_clock_time = msg_split[3]
        print("current trial: ", trial)
        print("clock time: ", clock_time, "wall clock time: ", wall_clock_time)

        results.append({
            "solver": "hypre",
            "threads": thread,
            "clock_time": clock_time,
            "wall_clock_time": wall_clock_time,
            "matrix_info": A,
            "trial": trial
        })

df = pd.DataFrame(results)

output_file = os.path.join(data_dir, "hypre2.csv")
df.to_csv(output_file, index=False)



