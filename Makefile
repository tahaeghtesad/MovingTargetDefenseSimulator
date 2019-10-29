default:
		@echo 'You once ruined all my experiments!' | cowsay
		@printf 'clean: cleans everything\nrun: runs the experiment based on run.py\nstop: cancel every running job\ncount: number of running jobs\nlist: list of running jobs\nwatch: watch job list\ncollect: run log collector\n'

clean:
		@printf 'Are you sure you wanna clean everything?\nI will wait for 15 seconds\nin case you changed your mind\n' | cowsay
		@sleep 15
		rm slurm* || rm -r tb_logs/ || rm -r logs/ || rm -r weights/ || rm -r reward_plots/
		mkdir tb_logs logs weights reward_plots
		git stash


run:
		#module load Anaconda3/python-3.6
		python run.py

stop:
		scancel -n MTD

count:
		squeue | grep MTD | wc -l

list:
		squeue -l | grep MTD

watch:
		watch -n 10 "squeue -l | grep MTD | sort -k1 -n"

collect:
		sbatch srun.collect.sh
