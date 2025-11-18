#-*- coding:utf-8 -*-
import os
import sys
import subprocess
import time
import re
import json
# cur_path = os.getcwd()
import psutil
import signal
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

def run_exp(solver, bin_path, data_dir, save_dir, json_file="filenames.json",skipwords=[],whitelist=[],timeout=30,nullspace=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    explog="%s/0sumup.log"%save_dir

    with open(json_file,"r")as f:
        filenames=json.load(f)
    prenames=filenames["prenames"]

    num1, num2, num3 = 0,0,0

    prename_need_run = []
    for prename in prenames:
        log_file="%s/%s.log"%(save_dir,prename)
        if os.path.exists(log_file):
            print(prename+" FILE EXIST, MIGHT BE DONE, NEED CHECK") # TODO, write a checker to evaluate if the log is fully filled.
            num1+=1
        else:
            print(prename+" FILE DOES NOT EXIST, NEED RUN")
            print("Reason:")

            current_folder = os.path.join(data_dir,prename)
            if os.path.exists(current_folder):
                files = [f for f in os.listdir(current_folder) if os.path.isfile(os.path.join(current_folder, f))]
                num_file = len(files)
                if num_file != 0:
                    print(f"Need to run, number of files in {current_folder}: {num_file}")
                    prename_need_run.append(prename)
                    num2+=1
                else:
                    print(f"Empty folder, number of files in {current_folder}: {num_file}")
                    num3+=1
            else:
                print(f"No such folder {current_folder}")
                num3+=1
    print("%d done, %d need run, %d empty or no folder \n"%(num1,num2,num3))

    for prename in prename_need_run:
        print("running", prename)
        skip=False
        for word in skipwords:
            if word in prename:
                skip=True
                break                
        if skip:
            print("Skip "+prename)
            continue

        if whitelist!=[]:
            skip=True
            for word in whitelist:
                if word in prename:
                    skip=False
                    break
            if skip:
                continue            

        log_file="%s/%s.log"%(save_dir,prename)
        start_i=0
        start_j=1
        end_i = 1000
        end_j = 1000

        if os.path.exists(log_file):
            with open(log_file,"r")as f:
                print(log_file)
                last_line=f.readlines()[-1]
                last_i=int(last_line.split(" ")[1])
                last_j=int(last_line.split(" ")[2])
            start_i=last_i+1
            start_j=last_j+1


        i = start_i
        j = start_j
        search_first_file = False
        for out_i in range(start_i,end_i):
            for in_j in range(start_j,end_j):
                fp_a=os.path.join(data_dir,prename,"%d_%d_A.bin"%(out_i,in_j))
                if os.path.exists(fp_a):
                    i = out_i
                    j = in_j
                    search_first_file = True
                    fp_b=os.path.join(data_dir,prename,"%d_%d_b.bin"%(out_i,in_j))
                    break
            if search_first_file:
                print("found")
                break
        # fp_a=os.path.join(data_dir,prename,"%d_%d_A.bin"%(i,j))
        # fp_b=os.path.join(data_dir,prename,"%d_%d_b.bin"%(i,j))


        while(os.path.exists(fp_a) and i < end_i):
            print("i:", i)
            while(os.path.exists(fp_a) and j < end_j):
                print("j: ", j)
                if nullspace:
                    fp_nullspace=os.path.join(data_dir,prename,"%d_%d_nullspace.bin"%(i,j))
                if nullspace:
                    cmd_string="%s %s %s %s %s"%(
                                                        bin_path,
                                                        fp_a,
                                                        fp_b,
                                                        fp_nullspace,
                                                        solver)
                else:
                    cmd_string="%s %s %s %s"%(
                                                        bin_path,
                                                        fp_a,
                                                        fp_b,
                                                        solver)                
                code,msg,mem=run_cmd(cmd_string, timeout*60)  # timeout=30*60sec
                with open(explog,"a")as f:
                    f.write("%s %d %s %f"%(cmd_string,code,msg,mem)+"\n")
                with open(log_file,"a")as f:
                    f.write("%s %d %d %d %s %f"%(prename,i,j,code,msg,mem)+"\n")
                j += 1
                fp_a=os.path.join(data_dir,prename,"%d_%d_A.bin"%(i,j))
                fp_b=os.path.join(data_dir,prename,"%d_%d_b.bin"%(i,j))
            j = 1
            i += 1
            fp_a=os.path.join(data_dir,prename,"%d_%d_A.bin"%(i,j))
            fp_b=os.path.join(data_dir,prename,"%d_%d_b.bin"%(i,j))


def main(argv):
    # timeout=int(argv[2])
    # date=argv[3]
    # solver=argv[1]

    timeout=20
    name="20250704_change_tolerance"

    os.environ['OMP_NUM_THREADS'] = "1"

    # for solver in ["AMGCL"]:
    #     run_exp(solver,
    #             bin_path="/u/1/chenyang/benchmark/build/TestMatLogger", 
    #             data_dir="/u/1/chenyang/benchmark_data/matrix_resource/solver-mat-0906", 
    #             save_dir="/u/1/chenyang/benchmark_data/exp-result/%s/%s"%(name, solver),  
    #             json_file="/u/1/chenyang/benchmark_data/matrix_resource/data-all/filenames.json",
    #             skipwords=[],
    #             whitelist=[],  # TODO: armadillo (not finished)
    #             timeout=timeout)
        
    for solver in ["Hypre"]:
        run_exp(solver,
                bin_path="/u/1/chenyang/benchmark/build/TestMatLogger", 
                data_dir="/u/1/chenyang/benchmark_data/matrix_resource/solver-mat-0906", 
                save_dir="/u/1/chenyang/benchmark_data/exp-result/%s/%s"%(name, solver),  
                json_file="/u/1/chenyang/benchmark_data/matrix_resource/data-all/filenames.json",
                skipwords=[],
                whitelist=[],  # TODO: armadillo (not finished)
                timeout=timeout)

    # for solver in ["Trilinos"]:
    #     run_exp(solver,
    #             bin_path="/home/yibo/myrepo/ExpPolySolve/build.ninja/ExpPolySolve_bin", 
    #             data_dir="/u/3/yibo/exp-result/0808/solver-mat", 
    #             save_dir="/u/3/yibo/exp-result/%s/%s"%(date, solver),  
    #             json_file="/home/yibo/myrepo/exp-data/data-all/filenames.json",
    #             skipwords=[],
    #             whitelist=[],
    #             timeout=timeout,
    #             nullspace=False)

    # for solver in ["Trilinos"]:
    #     run_exp(solver,
    #             bin_path="/home/yibo/myrepo/ExpPolySolve/build.ninja/ExpPolySolve_bin", 
    #             data_dir="/home/yibo/myrepo/matrix_resource/solver-mat-0906", 
    #             save_dir="/home/yibo/myrepo/exp-result/%s/%s"%(date, solver),  
    #             json_file="/home/yibo/myrepo/matrix_resource/data-all/filenames.json",
    #             skipwords=[],
    #             whitelist=[], 
    #             timeout=timeout,
    #             nullspace=False)
    
        
    # for solver in ["Trilinos-nullspace"]:
    #     run_exp(solver,
    #             bin_path="/home/yibo/myrepo/ExpPolySolve/build.ninja/ExpPolySolve_bin", 
    #             data_dir="/u/3/yibo/exp-result/0808/solver-mat", 
    #             save_dir="/u/3/yibo/exp-result/%s/%s"%(date, solver),  
    #             json_file="/home/yibo/myrepo/exp-data/data-all/filenames.json",
    #             skipwords=[],
    #             whitelist=[],
    #             timeout=timeout,
    #             nullspace=True)
        
    # os.environ['OMP_NUM_THREADS'] = "1"
    # for solver in ["Eigen::PardisoLDLT"]:
    #     run_exp(solver,
    #             bin_path="/home/yibo/myrepo/ExpPolySolve/build.chenyang/TestMatTime", 
    #             data_dir="/home/yibo/myrepo/matrix_resource/solver-mat-0906", 
    #             save_dir="/home/yibo/myrepo/exp-result/%s/%s"%(date, solver),  
    #             json_file="/home/yibo/myrepo/matrix_resource/data-all/filenames.json",
    #             skipwords=[],
    #             whitelist=[], 
    #             timeout=timeout)


if __name__ == "__main__":
    main(sys.argv)

