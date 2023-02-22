export mismip ?= .
export OMP_NUM_THREADS=1

SCRIPT := $(mismip)/uniform.py

$(mismip)/steady-state.h5: $(SCRIPT) 
	mpirun -n 8 python $(SCRIPT) \
		--output $@ \
		--half \
		--resolution 500 \
		--simulation-end-time 12200.0 \
		--simulation-timestep 2.0 \
		--timesteps-per-export 100
		
# $(mismip)/retreated.h5: $(SCRIPT) 
# 	mpirun -n 8 python $(SCRIPT) \
# 		--output $@ \
# 		--half \
# 		--input steady-state.h5 \
# 		--resolution 1000 \
# 		--simulation-end-time 102.0 \
# 		--simulation-timestep 0.04 \
# 		--timesteps-per-export 10 \
# 		--melt \
# 		--input-idx 56

# $(mismip)/readvanced.h5: $(SCRIPT) 
# 	mpirun -n 8 python $(SCRIPT) \
# 		--output $@ \
# 		--half \
# 		--input retreated.h5 \
# 		--resolution 1000 \
# 		--simulation-end-time 102.0 \
# 		--simulation-timestep 0.04 \
# 		--timesteps-per-export 10 \
# 		--input-idx 250
