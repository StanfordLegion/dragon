# Copyright (c) 2014-2015 Los Alamos National Security, LLC
#                         All rights reserved.
#
# This file is part of the  LANL Contributions to Legion (C15091) project.
# See the LICENSE.txt file at the top-level directory of this distribution.

#CUDA ?=/usr/local/cuda-7.0
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif
NO_BUILD_RULES=1
#USE_CUDA=1
#Flags for directing the runtime makefile what to include
#DEBUG           ?= 1		# Include debugging symbols
#OUTPUT_LEVEL    ?= 2	# Compile time print level
#SHARED_LOWLEVEL ?= 0		# Use the shared low level
#ALT_MAPPERS     ?= 0		# Compile the alternative mappers

# Put the binary file name here
OUTFILE		= libDragon.a
# List all the application source files here
GEN_SRC		= 	SingleKernelLauncher.cc LRWrapper.cc LPWrapper.cc \
				IndexKernelLauncher.cc  BetterMapper.cc \
				MustEpochKernelLauncher.cc #dim3_sweep_interface.cc# .cc files
GEN_GPU_SRC	=  	# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
#INC_FLAGS	?=    #-I/home/payne/Development/FieldKernelOp
#CC_FLAGS	?=    -g   -std=c++11 -DLEGION_PROF -DORDERED_LOGGING -DDYNAMIC_TESTS -DNODE_LOGGING #-DDEBUG_HIGH_LEVEL
#NVCC_FLAGS	?=    -g -G   -std=c++11 -DLEGION_PROF -DORDERED_LOGGING -DDYNAMIC_TESTS -DNODE_LOGGING#-DDEBUG_HIGH_LEVEL
#GASNET_FLAGS	?=
#LD_FLAGS	?=  -lboost_system -lboost_program_options 








###########################################################################
#
#   Don't change anything below here
#   
###########################################################################



include $(LG_RT_DIR)/runtime.mk



# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm -f
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
ifndef GCC
GCC	:= g++
endif
ifndef NVCC
NVCC	:= nvcc
endif
SSH	:= ssh
SCP	:= scp


GEN_OBJS	:= $(GEN_SRC:.cc=.o)
# Only compile the gpu objects if we need to 
ifeq ($(strip $(SHARED_LOWLEVEL)),0)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:=
endif

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS)

.PHONY: all
all: $(OUTFILE)

CC_FLAGS+=-fPIC

# If we're using the general low-level runtime we have to link with nvcc
#$(OUTFILE) : $(ALL_OBJS)
#	@echo "---> Linking objects into one binary: $(OUTFILE)"
#	ld -r $(ALL_OBJS) -L$(BOOST_LIBS) -lboost_system -lboost_program_options  -o $(OUTFILE) 

$(OUTFILE) : $(ALL_OBJS)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(NVCC) -g -G  --lib $(ALL_OBJS) -o $(OUTFILE) 


$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
	
%.o : %.cpp
	$(CXX) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)	 -fPIC

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

ifeq ($(strip $(SHARED_LOWLEVEL)),0)
$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS) -DUSE_CUDA_FKOP
else
$(GEN_GPU_OBJS) : %.o : %.cu
	$(GCC) -o $@ -x c++ -c $<  $(INC_FLAGS) $(CC_FLAGS)
endif

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)
