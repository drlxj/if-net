import os
import time
import datetime

if __name__ == "__main__":
    #os.system("conda activate if-net")
    #time_start = 0
    #submitted = True
    #running = True
    queue_start = False
    queue_start_time = 0
    current_job_str = ""
    kill_send = False
    while True:
        f = os.popen("bbjobs")
        #f = os.popen("bsub -n 8 -W 4:00 -R \"rusage[mem=8192, ngpus_excl_p=4]\" -G ls_polle_s ./train.sh")
        s = f.readlines()
        if len(s) == 1:
            t0 = time.time()
            print("Try to submit new task at "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            result = os.system("bsub -n 8 -W 4:00 -R \"rusage[mem=8192, ngpus_excl_p=4]\" -G ls_polle_s ./train.sh")
            #bsubs = f.readlines()
            #result = bsubs[1][19:28] == "submitted" 
            #result = os.system("./aggresive_train.sh")
            if not result:
                #current_job_str = bsubs[1][5:14]
                kill_send = False
                print(f"Submission successed after {time.time()-t0:5.2f}s")
            else:
                print("Submission failed, exit")
                exit(1)
        #elif s[16][2:9] == "Started":
        elif len(s)>25:
            time_str = s[16][40:56]
            relative_time = 60*(int(time_str[3:5])+60*int(time_str[0:2]))
            date = datetime.datetime.strptime(time_str[6:],"%Y-%m-%d")
            day_time = datetime.datetime.timestamp(date)
            start_time = day_time + relative_time
            #print(f"time.time:{time.time()}   time_calculated:{current_time}")
            time_to_finish = -time.time() + start_time + 4*3600 - 5*60
            #time_to_finish = -time.time() + start_time + 60*2
            current_job_str = s[1][40:49]
            if queue_start:
                queue_start = False
                print(f"Start running after queueing for {int(time.time()-queue_start_time)}s")
            if time_to_finish<30:
                if not kill_send:
                    print("In order to save log file correctly, kill the bsub before time out")
                    os.system(f"bkill {current_job_str}")
                    kill_send = True
                time.sleep(2)
            else:
                if time_to_finish>15*60:
                    time_to_finish = 15*60
                wake_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + time_to_finish))
                print(f"Still a long time to wait, sleep and will wake at {wake_time}")
                time.sleep(time_to_finish)
        else:
            if not queue_start:
                queue_start = True
                queue_start_time = time.time()
            time.sleep(2)

