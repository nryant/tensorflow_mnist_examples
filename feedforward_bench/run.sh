#!/bin/bash

# Benchmark using official 0.6.0 release.
#PYTHONPATH=../:$PYTHONPATH ./bench.py 0.6.0_timings.txt &> 0.6.0.log

# Benchmark using b1cabed4e60015602dacd66ea39d419db50c3e1b
#PYTHONPATH=../:$PYTHONPATH ./bench.py b1cabed_timings.txt &> b1cabed.log

# Benchmark using d1b8333effdcb031e6e34a2835a2f1c877fdd79b
#PYTHONPATH=../:$PYTHONPATH ./bench.py d1b8333_timings.txt &> d1b8333.log

# Benchmark using 039981f5a382ce9dc1e97dc3bd25aeba7fd82ade
PYTHONPATH=../:$PYTHONPATH ./bench.py 039981f_timings.txt &> 039981f.log
