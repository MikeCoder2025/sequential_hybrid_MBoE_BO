## Repository Overview

This repository contains five files, four of which are Python scripts implementing the following experimental optimisation strategies:

1. Bayesian Optimisation (BO)
2. Goal-Oriented Optimal Experimental Design (GOOED)
3. Model-Based Design of Experiments (MBDoE)
4. Sequential Hybrid MBDoE–BO

Each Python script is configured to execute the complete optimisation workflow **three times**. A single run represents one complete execution of the corresponding method from initialization to termination.

The three independent runs are used to calculate the **mean computational time** and assess the run-to-run variation in computational performance. These results enable a consistent comparison of the computational requirements of BO, GOOED, MBDoE, and the sequential hybrid MBDoE–BO framework.

## Running the Codes

Clone the repository and navigate to the repository directory in a terminal.

The individual methods can then be executed using:

```bash
python BO_Standalone_3Run_MeanRuntime.py
```

```bash
python MBDoE_Standalone_3Run_MeanRuntime.py
```

```bash
python GOOED_Standalone_3Run_MeanRuntime.py
```

```bash
python Sequential_Hybrid_MBDoEBO_EI_Standalone_3Run_MeanRuntime.py
```

Each script will execute the corresponding methodology three times and report the computational-time results for comparison across the four experimental optimisation strategies.
