/*
 *  A simple implementation of a printf-style CUDA-kernel print function.
 *  Copyright (C) 2011-2014  Guido Klingbeil
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
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>


/*
 * These magic numbers may be changed by the user.
 * There is no locking mechanism between the host and GPU,
 * so these numbers need. The chosen values seem to work fine though.
 */
#define BUFFERLENGTH 1024
#define MAXSTRINGLENGTH 1024


/*
 * How often does the CPU thread flush the GPU buffer to screen or file
 */
#define INTERVAL_SEC 0
#define INTERVAL_USEC 1000000
//#define INTERVAL_USEC 100000000

/*
 * Default precision is 4 digits
 */
#define DEFAULTDIGITS 4


/*
 * Do not change. The maximum 64-bit integer value is
 * 2^64 - 1 = 18,446,744,073,709,551,615. Set it to 18 digits
 * to avoid overflow.
 */
#define MAXINTDIGITS 25


/*
 * Do not change. The number of buffers depends
 *  on the integer data type (uint8_t) used.
 */
#define STREAMS 256


typedef struct {
	uint32_t endPtr;
	uint8_t chars[BUFFERLENGTH];
} printBuffer;


/*
 * Global array of pointers to buffer structs:
 * printBuffer[0] stdout
 * printBuffer[1] stderr
 * printBuffer[2] logfile0
 * printBuffer[3] logfile1
 * ...
 * printBuffer[255] logfile253
 */
__device__ printBuffer *dBuffers[STREAMS];


/*
 * Global variables
 */
uint8_t done = 0;
uint8_t _inited = 0;
printBuffer *hBuffers[STREAMS];
uint32_t startPtr[STREAMS];
pthread_t wThread;
FILE *outStreams[STREAMS];
pthread_mutex_t runMutex = PTHREAD_MUTEX_INITIALIZER;


/*
 * A tiny CUDA kernel with the only purpose to write the
 * address of the buffer to a global variable on the device.
 */
__global__ void setDevicePtr(printBuffer *bufferPtr, uint8_t gpuStream) {
	dBuffers[gpuStream] = bufferPtr + gpuStream;
	return;
}


/*
 * Copy message to the pinned buffer
 */
__device__ uint32_t writeToStream(char *string, uint32_t stringLength, uint8_t gpuStream) {
	uint32_t startPtr = 0, i = 0;

	// use atomic add to set the endPtr
	startPtr = atomicAdd(&(dBuffers[gpuStream] -> endPtr), stringLength);

	for (i = 0; i < stringLength; i++) {
		dBuffers[gpuStream] -> chars[(startPtr + i) % BUFFERLENGTH] = string[i];
	}

	return (stringLength);
}


/*
 * Convert a uint64 to a string.
 */
__device__ uint32_t parseUint64(char *outBuffer, uint64_t u) {
    int64_t digits = 1, position = 0, tmp = u;

	if (u == 0) {
		// the char 0 has the ASCII code 48
		outBuffer[0] = 48;
		return (1);
	}

	// get the number of digits using binary search
    if( tmp >= 1000000000 ) {
        digits *= 1000000000 ;
        tmp /= 1000000000;
    }
    
    if( tmp >= 100000) {
        digits *= 100000;
        tmp /= 100000;
    }
    
    if( tmp >= 1000) {
        digits *= 1000;
        tmp /= 1000;
    }

    if( tmp >= 100) {
        digits *= 100;
        tmp /= 100;
    }

     if( tmp >= 10) {
        digits *= 10;
        tmp /= 10;
    }

	// start parsing with the leftmost digit
    // the char 0 has the ASCII code 48
	while (digits > 0) {
		outBuffer[position++] = 48 + ((u / digits) % 10);
		digits /= 10;
	}

	return (position);
}


/*
 * If we detect a NAN or INF, we simply write it to screen.
 */
__device__ uint32_t setNAN(char *outBuffer) {
	outBuffer[0] = 'N';
	outBuffer[1] = 'A';
	outBuffer[2] = 'N';
	return (3);
}


__device__ uint32_t setINF(char *outBuffer) {
	outBuffer[0] = 'I';
	outBuffer[1] = 'N';
	outBuffer[2] = 'F';
	return (3);
}


/*
 * Convert an int64_t to a string.
 */
__device__ uint32_t parseInt64(char *outBuffer, int64_t i) {
	if (i < 0) {
		outBuffer[0] = '-';

        // corner case for min int64 number
        // since we can only parse positive numbers
        if(i == -9223372036854775808) return( setINF(outBuffer + 1) + 1);

		return (parseUint64(outBuffer + 1, abs(i)) + 1);
	} else {
		return (parseUint64(outBuffer, i));
	}
}


/*
 * Parse a double value in standard notation
 */
__device__ uint32_t parseDouble(char *outBuffer, double f, uint16_t precision) {
	uint32_t position = 0;
	double u;

	// check for NAN and INF
	if (isnan(f)) {
	    return(setNAN(outBuffer));
	}

	if (isinf(f)) {
		return(setINF(outBuffer));
	}

	position += parseInt64(outBuffer, (int64_t) f);
	outBuffer[position++] = '.';

	// apply mathematical correct rounding
	u = (abs(f - (int64_t)f) * exp10((double)precision));
	if(	u - (uint64_t)(u) >= 0.5f) {
		position += parseUint64(outBuffer + position, (uint64_t)(u + 1));
	} else {
		position += parseUint64(outBuffer + position, (uint64_t)(u));
	}

	return (position);
}


/*
 * Parse a double value in scientific notation. The precision does
 * only effect the mantissa, not the exponent.
 */
__device__ uint32_t parseSciDouble(char *outBuffer, double f, uint16_t precision) {
	uint32_t position = 0;
	int64_t exponent = 0;
	double mantissa = 0.0;

	// check for NAN and INF
	if (isnan(f)) {
		return(setNAN(outBuffer));
	}

	if (isinf(f)) {
		return(setINF(outBuffer));
	}

	// get the exponent
	exponent = (int32_t)log10(fabs(f));

	// get the significand or mantissa
	mantissa = f / exp10((double)exponent);

	// normalise the mantissa
	while(abs(mantissa) < 1.0) {
		exponent --;
		mantissa = f / exp10((double)exponent);
	}

	while(abs(mantissa) > 10.0) {
		exponent ++;
		mantissa = f / exp10((double)exponent);
	}

	position += parseDouble(outBuffer + position, mantissa, precision);
	outBuffer[position++] = 'e';
	position += parseInt64(outBuffer + position, exponent);

	return (position);
}


/*
 * Extract two characters of precision. Note, there is no error checking if more
 * precision characters are given.
 */
__device__ uint16_t precision(char * inBuffer, uint16_t *numDigits) {
    uint16_t i = 0;

    if(inBuffer[1] == '.') {
	    if(inBuffer[2] - 48 < 10 && inBuffer[2] - 48 >= 0) {
		    *numDigits = inBuffer[2] - 48;
			i += 2;
		}

	    if(inBuffer[i + 1] - 48 < 10 && inBuffer[i + 1] - 48 >= 0) {
			*numDigits = 10* *numDigits + inBuffer[i + 1] - 48;
			i ++;
		}

	    if(*numDigits == 0) *numDigits = 1;
	    if(*numDigits > MAXINTDIGITS) *numDigits = MAXINTDIGITS;
    }

    return (i);
}


/*
 *  Templates are not able to exclude certain data types.
 *  Rather than excluding char *, we use an empty template and
 *  specialisations to catch all the data types we want to print.
 *  Not particularly elegant, but it works.
 */
__device__ void format(char *inBuffer, char *outBuffer, uint16_t *iPosition, uint16_t *oPosition) {
    uint16_t i = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

        outBuffer[(*oPosition)++] = inBuffer[i];
	}
	return;
}


template <class TYPE> __device__ void format(char *inBuffer, char *outBuffer, TYPE t, uint16_t *iPosition, uint16_t *oPosition) {
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, char* s, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0, j = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			if(inBuffer[i + 1] == 's') {
				// assume a string can be max 1024 char in length
				for(j = 0; s[j] && j < MAXSTRINGLENGTH; j ++) {
					outBuffer[(*oPosition)++] = s[j];
				}
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, char t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			if(inBuffer[i + 1] == 'c') {
				outBuffer[(*oPosition)++] = (char)t;
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, uint64_t t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			if(inBuffer[i + 1] == 'c') {
				outBuffer[(*oPosition)++] = (char)t;
			} else if(inBuffer[i + 1] == 'd') {
				(*oPosition) += parseInt64(outBuffer + (*oPosition), (int64_t)t);
			} else if(inBuffer[i + 1] == 'u') {
				(*oPosition) += parseUint64(outBuffer + (*oPosition), (uint64_t)t);
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, unsigned int t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			if(inBuffer[i + 1] == 'c') {
				outBuffer[(*oPosition)++] = (char)t;
			} else if(inBuffer[i + 1] == 'd') {
				(*oPosition) += parseInt64(outBuffer + (*oPosition), (int64_t)t);
			} else if(inBuffer[i + 1] == 'u') {
				(*oPosition) += parseUint64(outBuffer + (*oPosition), (uint64_t)t);
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, int t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			if(inBuffer[i + 1] == 'c') {
				outBuffer[(*oPosition)++] = (char)t;
			} else if(inBuffer[i + 1] == 'd') {
				(*oPosition) += parseInt64(outBuffer + (*oPosition), (int64_t)t);
			} else if(inBuffer[i + 1] == 'u') {
				(*oPosition) += parseUint64(outBuffer + (*oPosition), (uint64_t)t);
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, int64_t t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			if(inBuffer[i + 1] == 'c') {
				outBuffer[(*oPosition)++] = (char)t;
			} else if(inBuffer[i + 1] == 'd') {
				(*oPosition) += parseInt64(outBuffer + (*oPosition), (int64_t)t);
			} else if(inBuffer[i + 1] == 'u') {
				(*oPosition) += parseUint64(outBuffer + (*oPosition), (uint64_t)t);
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, float t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;
	uint16_t minDigits = DEFAULTDIGITS;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			i += precision(&inBuffer[i], &minDigits);

			if(inBuffer[i + 1] == 'f') {
				(*oPosition) += parseDouble(outBuffer + (*oPosition), (double)t, minDigits);
			} else if(inBuffer[i + 1] == 'e') {
				(*oPosition) += parseSciDouble(outBuffer + (*oPosition), (double)t, minDigits);
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}


template <> __device__ void format<>(char *inBuffer, char *outBuffer, double t, uint16_t *iPosition, uint16_t *oPosition) {
	uint16_t i = 0;
	uint16_t minDigits = DEFAULTDIGITS;

	for(i = (*iPosition); inBuffer[i]; i ++ ) {
		if((*oPosition) >= BUFFERLENGTH - 1) {
			(*iPosition) = i;
			break;
		}

		if(inBuffer[i] != '%') {
			outBuffer[(*oPosition)++] = inBuffer[i];
		} else {
			i += precision(&inBuffer[i], &minDigits);

			if(inBuffer[i + 1] == 'f') {
				(*oPosition) += parseDouble(outBuffer + (*oPosition), (double)t, minDigits);
			} else if(inBuffer[i + 1] == 'e') {
				(*oPosition) += parseSciDouble(outBuffer + (*oPosition), (double)t, minDigits);
			}
			i += 2;
			(*iPosition) = i;
			break;
		}
	}
	return;
}

__device__ uint32_t dPrintf(uint8_t gpuStream, char* s) {
    char tmpBuffer[BUFFERLENGTH + MAXSTRINGLENGTH];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}

template <class TYPE> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE t) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	// assume a string length of 1024 chars
	char tmpBuffer[BUFFERLENGTH + MAXSTRINGLENGTH];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4, class TYPE5> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4, TYPE5 t5) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);
	format(s, (char *)tmpBuffer, t5, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4, class TYPE5, class TYPE6> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4, TYPE5 t5, TYPE6 t6) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);
	format(s, (char *)tmpBuffer, t5, &i, &position);
	format(s, (char *)tmpBuffer, t6, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4, class TYPE5, class TYPE6, class TYPE7> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4, TYPE5 t5, TYPE6 t6, TYPE7 t7) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);
	format(s, (char *)tmpBuffer, t5, &i, &position);
	format(s, (char *)tmpBuffer, t6, &i, &position);
	format(s, (char *)tmpBuffer, t7, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4, class TYPE5, class TYPE6, class TYPE7, class TYPE8> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4, TYPE5 t5, TYPE6 t6, TYPE7 t7, TYPE8 t8) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);
	format(s, (char *)tmpBuffer, t5, &i, &position);
	format(s, (char *)tmpBuffer, t6, &i, &position);
	format(s, (char *)tmpBuffer, t7, &i, &position);
	format(s, (char *)tmpBuffer, t8, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4, class TYPE5, class TYPE6, class TYPE7, class TYPE8, class TYPE9> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4, TYPE5 t5, TYPE6 t6, TYPE7 t7, TYPE8 t8, TYPE9 t9) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);
	format(s, (char *)tmpBuffer, t5, &i, &position);
	format(s, (char *)tmpBuffer, t6, &i, &position);
	format(s, (char *)tmpBuffer, t7, &i, &position);
	format(s, (char *)tmpBuffer, t8, &i, &position);
	format(s, (char *)tmpBuffer, t9, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


template <class TYPE1, class TYPE2, class TYPE3, class TYPE4, class TYPE5, class TYPE6, class TYPE7, class TYPE8, class TYPE9, class TYPE10> __device__ uint32_t dPrintf(uint8_t gpuStream, char* s, TYPE1 t1, TYPE2 t2, TYPE3 t3, TYPE4 t4, TYPE5 t5, TYPE6 t6, TYPE7 t7, TYPE8 t8, TYPE9 t9, TYPE10 t10) {
	// a 64-bit integer is 20 digits long, our GPU buffer is 4096 chars.
	// To prevent access beyond the buffer, we need to allocate 4116 chars.
	char tmpBuffer[BUFFERLENGTH + 20];
	uint16_t i = 0, j = 0, position = 0;

	format(s, (char *)tmpBuffer, t1, &i, &position);
	format(s, (char *)tmpBuffer, t2, &i, &position);
	format(s, (char *)tmpBuffer, t3, &i, &position);
	format(s, (char *)tmpBuffer, t4, &i, &position);
	format(s, (char *)tmpBuffer, t5, &i, &position);
	format(s, (char *)tmpBuffer, t6, &i, &position);
	format(s, (char *)tmpBuffer, t7, &i, &position);
	format(s, (char *)tmpBuffer, t8, &i, &position);
	format(s, (char *)tmpBuffer, t9, &i, &position);
	format(s, (char *)tmpBuffer, t10, &i, &position);

	for(j = i; s[j] && position < BUFFERLENGTH; j ++) {
		tmpBuffer[position++] = s[j];
	}

	// make sure we terminate the buffered string with 0
	tmpBuffer[BUFFERLENGTH - 1] = 0;

	return( writeToStream(tmpBuffer, position, gpuStream));
}


// the function executed by the thread
// We implement a cyclical buffer. The writing thread sets
// the value of endPtr to the end of the printed string.
void *writerThread(void *args) {
	uint32_t endPtr, _done = 0, j = 0, position = 0;
	FILE *outStream;

	// structures required by nanosleep
	timespec sleepFor, remaining;
	sleepFor.tv_sec = INTERVAL_SEC;
	sleepFor.tv_nsec = INTERVAL_USEC;

	while (!_done) {

		for (position = 0; position < STREAMS; position++) {
			outStream = outStreams[position];

			if (outStream != NULL) {

				// Flush the pinned buffer
				// Do nothing if start and end position are equal.
				// We potentially miss messages of exactly the buffer length.
				endPtr = (hBuffers[position] -> endPtr) % BUFFERLENGTH;

				//if (startPtr[position] != endPtr) {
				if (startPtr[position] <= endPtr) {
					// no cycling around necessary
					for (j = startPtr[position]; j < endPtr; j++) {
						fprintf(outStream, "%c", hBuffers[position] -> chars[j]);
					}
				} else {
					// towards the end of the buffer
					for (j = startPtr[position]; j < BUFFERLENGTH; j++) {
						fprintf(outStream, "%c", hBuffers[position] -> chars[j]);
					}

					// cycle around
					for (j = 0; j < endPtr; j++) {
						fprintf(outStream, "%c", hBuffers[position] -> chars[j]);
					}
				}
			    //}
				startPtr[position] = endPtr;
			}
		}

		// sleep
		nanosleep(&sleepFor, &remaining);

		// we are only reading, so no lock required
		pthread_mutex_lock(&runMutex);
		_done = done;
		pthread_mutex_unlock(&runMutex);
	}

	return NULL;
}


/*
 * Init and close the streams
 */

// initialises stdout and stderr only
int8_t initOStream() {

	printBuffer *dPtr;

	if (!_inited) {
		memset(startPtr, 0, STREAMS * sizeof(uint32_t));
		memset(outStreams, 0, STREAMS * sizeof(uint32_t));
		cudaSafeCall( cudaSetDeviceFlags(cudaDeviceMapHost));
		cudaSafeCall(cudaHostAlloc((void**) &hBuffers[0], STREAMS
				* sizeof(printBuffer), cudaHostAllocMapped));
		memset(hBuffers[0], 0, STREAMS * sizeof(printBuffer));
		_inited = 1;
	}
	outStreams[0] = stdout;
	outStreams[1] = stderr;

	//hBuffers[0] = buffer;
	cudaSafeCall(cudaHostGetDevicePointer((void**) &dPtr, hBuffers[0], 0));
	setDevicePtr<<<1,1>>>(dPtr, 0);
	cudaThreadSynchronize();
	hBuffers[1] = hBuffers[0] + 1;
	setDevicePtr<<<1,1>>>(dPtr, 1);
	cudaThreadSynchronize();

	pthread_create(&wThread, NULL, writerThread, NULL);

	return (0);
}


// initialises writing to a log file or any other user-defined stream
// the programmer has to keep track of the GPU buffer used
int8_t initFStream(FILE *outStream, uint8_t position) {
	printBuffer *dPtr;

	if (!_inited) {
		memset(startPtr, 0, STREAMS * sizeof(uint32_t));
		memset(outStreams, 0, STREAMS * sizeof(uint32_t));
		cudaSafeCall( cudaSetDeviceFlags(cudaDeviceMapHost));
		cudaSafeCall(cudaHostAlloc((void**) &hBuffers[0], STREAMS
				* sizeof(printBuffer), cudaHostAllocMapped));
		memset(hBuffers[0], 0, STREAMS * sizeof(printBuffer));
		_inited = 1;
	}

	if (position < 2) {
		fprintf(stderr, "initFStream: stream 0 and 1 are reserved.\n");
		return (-1);
	}

	if (outStreams[position] != NULL) {
		fprintf(stderr, "initFStream: stream already in use.\n");
		return (-1);
	}

	hBuffers[position] = hBuffers[0] + position;
	cudaSafeCall(cudaHostGetDevicePointer((void**) &dPtr, hBuffers[0], 0));
	setDevicePtr<<<1,1>>>(dPtr, position);
	cudaThreadSynchronize();

	pthread_mutex_lock(&runMutex);
	outStreams[position] = outStream;
	pthread_mutex_unlock(&runMutex);

	return (0);
}


uint8_t closeStreams() {
	pthread_mutex_lock(&runMutex);
	done = 1;
	pthread_mutex_unlock(&runMutex);

	// wait for the writer thread to join
	pthread_join(wThread, NULL);

	// free page locked memory
    cudaSafeCall(cudaFreeHost(hBuffers[0]));

	return (0);
}

