export mismip ?= .
export OMP_NUM_THREADS=1

SCRIPT := $(mismip)/uniform.py

# $(mismip)/steady-state-0.h5: $(SCRIPT) 
# 	mpirun -n 16 python $(SCRIPT) \
# 		--output $@ \
# 		--output-level 0 \
# 		--simulation-end-time 6e3 \
# 		--simulation-timestep 5.0 \
# 		--timesteps-per-export 100

$(mismip)/steady-state-1.h5: $(SCRIPT) 
	mpirun -n 16 python $(SCRIPT) \
		--output $@ \
		--output-level 1 \
		--input steady-state-0.h5 \
		--input-idx 12 \
		--input-level 0 \
		--simulation-end-time 3.6e3 \
		--simulation-timestep 2.0 \
		--timesteps-per-export 100 

$(mismip)/steady-state-2.h5: $(SCRIPT) 
	mpirun -n 16 python $(SCRIPT) \
		--output $@ \
		--output-level 2 \
		--input steady-state-1.h5 \
		--input-level 1 \
		--simulation-end-time 1e3 \
		--simulation-timestep 1.0 \
		--timesteps-per-export 100

$(mismip)/steady-state.h5: $(SCRIPT) 
	mpirun -n 16 python $(SCRIPT) \
		--output $@ \
		--output-level 3 \
		--input steady-state-2.h5 \
		--input-level 2 \
		--simulation-end-time 200 \
		--simulation-timestep 1.0 \
		--timesteps-per-export 10