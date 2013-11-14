"""
From parakeet project.

https://github.com/iskandr/parakeet

License included:
New BSD License

Copyright (c) 2013 - Alex Rubinsteyn.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission. 


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.



"""
import numpy as np

# partial mapping, since ctypes doesn't support
# complex numbers
bool8 = np.dtype('bool8')

int8 = np.dtype('int8')
uint8 = np.dtype('uint8')

int16 = np.dtype('int16')
uint16 = np.dtype('uint16')

int32 = np.dtype('int32')
uint32 = np.dtype('uint32')

int64 = np.dtype('int64')
uint64 = np.dtype('uint64')
  
float32 = np.dtype('float32')
float64 = np.dtype('float64')

complex64 = np.dtype('complex64')
complex128 = np.dtype('complex128')

#for mapping dtypes to integers
nd_list = [
bool8,
int8,
uint8,
int16,
uint16,
int32,
uint32, 
int64,
uint64,
float32,
float64,
complex64,
complex128]

nd_dict = dict([(tpe,i) for i,tpe in enumerate(nd_list)])
  
  
def is_float(dtype):
  return dtype.type in np.sctypes['float']

def is_signed(dtype):
  return dtype.type in np.sctypes['int']
  
def is_unsigned(dtype):
  return dtype.type in np.sctypes['uint']

def is_complex(dtype):
  return dtype.type in np.sctypes['complex']
   
def is_bool(dtype):
  return dtype == np.bool8
   
def is_int(dtype):
  return is_bool(dtype) or is_signed(dtype) or is_unsigned(dtype)
  

import ctypes 
_to_ctypes = {
  bool8 : ctypes.c_bool, 
  
  int8  : ctypes.c_int8, 
  uint8 : ctypes.c_uint8, 
  
  int16 : ctypes.c_int16, 
  uint16 : ctypes.c_uint16,
  
  int32 : ctypes.c_int32, 
  uint32 : ctypes.c_uint32,
  
  int64 : ctypes.c_int64, 
  uint64 : ctypes.c_uint64, 
  
  float32 : ctypes.c_float, 
  float64 :  ctypes.c_double, 
}

def to_index(dtype):
    if dtype in nd_dict:
        return nd_dict[dtype]
    elif np.dtype(dtype) in nd_dict:
        return nd_dict[np.dtype(dtype)]
    else:
        raise RuntimeError("%s not found" %dtype)


def to_ctypes(dtype):
  """
  Give the ctypes representation for each numpy scalar type. 
  Beware that complex numbers have no assumed representation 
  and thus aren't valid arguments to this function. 
  """
  if dtype in _to_ctypes:
    return _to_ctypes[dtype]
  elif np.dtype(dtype) in _to_ctypes:
    return _to_ctypes[np.dtype(dtype)]
  else:
    raise RuntimeError("No conversion from %s to ctypes" % dtype)
  
