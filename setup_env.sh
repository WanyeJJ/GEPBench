#!/bin/bash

# PETSc and SLEPc Environment Setup
export PETSC_DIR=/usr/local/petsc
export SLEPC_DIR=/usr/local/slepc
export LD_LIBRARY_PATH=/usr/local/petsc/lib:/usr/local/slepc/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/petsc/lib/pkgconfig:/usr/local/slepc/lib/pkgconfig:$PKG_CONFIG_PATH

# MPI settings for running as root (if needed)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Compiler flags
export PETSC_CC=mpicc
export PETSC_CXX=mpicxx
export PETSC_FC=mpif90

echo "PETSc and SLEPc environment variables set:"
echo "PETSC_DIR=$PETSC_DIR"
echo "SLEPC_DIR=$SLEPC_DIR"
echo "LD_LIBRARY_PATH includes PETSc and SLEPc libraries"
echo ""
echo "To compile PETSc/SLEPc programs, use:"
echo "mpicc -I/usr/local/petsc/include -I/usr/local/slepc/include -o program program.c -L/usr/local/petsc/lib -L/usr/local/slepc/lib -lpetsc -lslepc -lm"
echo ""
echo "To run MPI programs:"
echo "mpirun -np <number_of_processes> ./program" 