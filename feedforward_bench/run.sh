#!/bin/bash

# Benchmark using official 0.6.0 release.
#PYTHONPATH=../:$PYTHONPATH ./bench.py 0.6.0_timings.txt &> 0.6.0.log

# Benchmark using b1cabed4e60015602dacd66ea39d419db50c3e1b
PYTHONPATH=../:$PYTHONPATH ./bench.py b1cabed_timings.txt &> b1cabed.log
