/*
  *  A simple implementation of a printf-style CUDA-kernel print function.
  *  Copyright (C) 2011-2012  Guido Klingbeil
  *
  *  This program is free software: you can redistribute it and/or modify
  *  it under the terms of the GNU General Public License as published by
  *  the Free Software Foundation, either version 3 of the License, or
  *  (at your option) any later version.
  *
  *  This program is distributed in the hope that it will be useful,
  *  but WITHOUT ANY WARRANTY; without even the implied warranty of
  *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  *  GNU General Public License for more details.
  *
  *  You should have received a copy of the GNU General Public License
  *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  */

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include "cutil_inline.h"
#include "kernelprintf.cu"


/*
 * A little test kernel supposed to be run in a 2x2 grid.
 * 
 */
__global__ void printKernel() {
	    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        int64_t i64 = 9223372036854775807;
        int64_t i64n = -9223372036854775807;
        int32_t i32 = 2147483647;
        int32_t i32n = -2147483648; 	


        double npi = -3.141592653589793238462643383279;
        float spnpi = 3.1415926535f;

        for(unsigned int j = 0; j < 5; j ++) {
            dPrintf(0, "Thread %d, iteration %d.\n", tid, j);

            if(tid == 0) {   
                dPrintf(2, "Thread %d, iteration %d:\ta string: %s\n", (unsigned int)tid, j,  (char *)"My little string!");
                dPrintf(2, "Thread %d, iteration %d:\ta char: %c\n", (int)threadIdx.x, j, 'A');
            } else if(tid == 1) {
                dPrintf(3, "Thread %d, iteration %d:\tmax, min signed int (int64_t): %d, %d\n", (uint32_t)tid, (uint32_t)j, i64, i64n); // integers do not understand precision
                dPrintf(3, "Thread %d, iteration %d:\tmax, min signed int (int32_t): %d, %d\n", (uint64_t)tid, j, i32, i32n);
            } else if(tid == 2) {
                dPrintf(4, "Thread %u, iteration %d:\tfirst digits of -pi (float): %.12f (please note: the expected precision should be 6 decimal places)\n", (int32_t)tid, j, spnpi);
                dPrintf(4, "Thread %u, iteration %d:\tfirst digits of -pi (double): %.12f (please note: the expected precision should be 6 decimal places)\n", (int64_t)tid, j, npi);
            } else if(tid == 3) {
                dPrintf(5, "Please note: the expected precision should be 6 decimal places.\n");
                dPrintf(5, "Thread %u, iteration %d:\tfirst digits of -1000*pi (float): %.12e (please note: the expected precision should be 6 decimal places)\n", tid, j, 100000.0f * spnpi);
                dPrintf(5, "Thread %u, iteration %d:\tfirst digits of -0.001*pi (double): %.12e(please note: the expected precision should be 6 decimal places)\n", tid, j, 0.001 * npi);
            }
        }
}


int main() {
        // open files
        FILE *logFile0 = fopen("string_character_test.txt", "w");
        FILE *logFile1 = fopen("integers.txt", "w");
        FILE *logFile2 = fopen("float.txt", "w");
        FILE *logFile3 = fopen("scientific_notation.txt", "w");

        // open a few custom GPU output streams
        // and hook them up to some files
        initFStream(logFile0, 2);
        initFStream(logFile1, 3);
        initFStream(logFile2, 4);
        initFStream(logFile3, 5);

        // initialise stdout and stderr
        initOStream();

        printKernel<<<2, 2>>>();
        cudaThreadSynchronize();

        sleep(1);

        // wind-down
        closeStreams();

        // close the files
        fclose(logFile0);
        fclose(logFile1);
        fclose(logFile2);
        fclose(logFile3);
        return (0);
}
