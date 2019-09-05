default:
		echo 'You once ruined all my experiments!' | cowsay

clean:
		rm slurm*
		rm -r tb_logs/
		rm -r logs/
		rm -r weights/
		rm -r reward_plots/
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