# Local Time Stepping Implementation

This folder contains the implementation of the LTS (Local Time Stepping) method applied on a 1D wave equation using FEM and a Leap-Frog scheme.

The implementation was part of a University assigment for the MATH0024-1 "Modelling with partial differential equations" course. The implementation is available in a personal repo : [MATH0024 - Local Time Stepping Leap-Frog for the wave equation in 1D](https://github.com/julienbrandoit/MATH0024---Local-Time-Stepping-Leap-Frog-for-the-wave-equation-in-1D). With the implementation, I provide a report with detailed mathematical formulations and analysis.

The method implemented comes from the paper "Energy Conserving Explicit Local Time Stepping for Second-Order Wave Equations".

## Overview

In the original repo, I conduct the following numerical experiments:
1. Stability criterion for the classical Leap-Frog scheme.
2. Stability criterion when using a LTS Leap-Frog scheme.

## Results

We can clearly see the power of LTS, which enables a gain in stability (a more flexible stability criterion) without excessively increasing the computation workload.

I provide some of the figures that are in my report.
1. refined_mesh_sol.pdf and regular_mesh_sol.pdf are based on the classical Leap-Frog scheme.
2. local_time_stepping_sol_fine.pdf and local_time_stepping_sol_coarse.pdf are based on the LTS Leap-Frog method. 

By comparing all the figures, one can get insight about the gain of stability.