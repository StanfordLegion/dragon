/*
 * wrapper_defines.h
 *
 *  Created on: Aug 6, 2015
 *      Author: payne
 */

#ifndef WRAPPER_DEFINES_H_
#define WRAPPER_DEFINES_H_

#include <legion.h>
#include <typeinfo>
#ifdef USE_CUDA_FKOP
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif
#include "legion_tasks.h"
#include <sstream>
#include <unistd.h>

#define MAX_LEGION_MATRIX_DIMS 8

#define MAX_LEGION_MATRIX_FIELDS 10



#endif /* WRAPPER_DEFINES_H_ */
