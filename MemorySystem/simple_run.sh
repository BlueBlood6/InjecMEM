python ./MemorySystem/inject_generate.py --memorypath ./MemorySystem/memory_base/run_1 --steps 0 --benign_total 60 --seed 20

python ./MemorySystem/inject_generate.py --memorypath ./MemorySystem/memory_base/run_1 --steps 1 --domain health

python ./MemorySystem/inject_generate.py --memorypath ./MemorySystem/memory_base/run_1 --steps 2 --domain health --n_eval 10 --noise_max 3 --seed 26