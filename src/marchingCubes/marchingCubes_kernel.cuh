/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MARCHING_CUBES_KERNEL_CUH_
#define _MARCHING_CUBES_KERNEL_CUH_

#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>  // includes for helper CUDA functions
#include <helper_math.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "MCdefines.h"
#include "MCtables.h"

namespace MC_cuda
{
  // textures containing look-up tables
  cudaTextureObject_t triTex;
  cudaTextureObject_t numVertsTex;

  // volume data
  cudaTextureObject_t volumeTex;

  ////////////////////////////////////
  // declarations of host functions //
  ////////////////////////////////////

  void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);

  void createVolumeTexture(uchar *d_volume, size_t buffSize);

  void destroyAllTextureObjects();

  void launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, uchar *volume,
                            uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                            float3 voxelSize, float isoValue);

  void launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied,
                            uint *voxelOccupiedScan, uint numVoxels);

  void launch_generateTriangles(
      dim3 grid, dim3 threads, float4 *pos, float4 *norm,
      uint *compactedVoxelArray, uint *numVertsScanned, uint3 gridSize,
      uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize, float isoValue,
      uint activeVoxels, uint maxVerts);

  void launch_generateTriangles2(
      dim3 grid, dim3 threads, float4 *pos, float4 *norm,
      uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
      uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize,
      float isoValue, uint activeVoxels, uint maxVerts);

  void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements);

  //////////////////////////////////////
  // declarations of device functions //
  //////////////////////////////////////

  // an interesting field function
  __device__ __forceinline__ 
  float tangle(float x, float y, float z);

  // evaluate field function at point
  __device__ __forceinline__ 
  float fieldFunc(float3 p);

  // evaluate field function at a point
  // returns value and gradient in float4
  __device__ __forceinline__ 
  float4 fieldFunc4(float3 p);

  // sample volume data set at a point
  __device__  __forceinline__ 
  float sampleVolume(cudaTextureObject_t volumeTex, uchar *data, uint3 p, uint3 gridSize);

  // compute position in 3d grid from 1d index
  // only works for power of 2 sizes
  __device__  __forceinline__ 
  uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask);

  // classify voxel based on number of vertices it will generate
  // one thread per voxel
  __global__ void classifyVoxel(uint *voxelVerts, uint *voxelOccupied, uchar *volume, uint3 gridSize,
                                uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels, float3 voxelSize,
                                float isoValue, cudaTextureObject_t numVertsTex, cudaTextureObject_t volumeTex);

  // compact voxel array
  __global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
                                uint *voxelOccupiedScan, uint numVoxels);

  // compute interpolated vertex along an edge
  __device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1);

  // compute interpolated vertex position and normal along an edge
  __device__ void vertexInterp2(float isolevel, float3 p0, float3 p1, float4 f0, float4 f1, float3 &p, float3 &n);

  // calculate triangle normal
  __device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2);

  // generate triangles for each voxel using marching cubes
  // interpolates normals from field function
  __global__ void generateTriangles(
      float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, uint3 gridSize, 
      uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize, float isoValue, uint activeVoxels, 
      uint maxVerts, cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex);

  // version that calculates flat surface normal for each triangle
  __global__ void generateTriangles2(
      float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
      uchar *volume, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
      float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts,
      cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,
      cudaTextureObject_t volumeTex);
}
#endif
