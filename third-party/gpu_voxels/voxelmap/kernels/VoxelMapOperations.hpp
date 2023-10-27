// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_HPP_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_HPP_INCLUDED

#include "VoxelMapOperations.h"
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxel/DistanceVoxel.hpp>

#include "VoxelMapOperationsPBA.hpp"

namespace gpu_voxels {
namespace voxelmap {

// do added constant memory

__constant__ uint32_t const_voxel_map_dim[3]; // voxel map dim x, y, z
__constant__ float const_voxel_side_length[1]; // voxel side length
__constant__ uint16_t const_depth_size[2]; // depth image size width, height
__constant__ uint16_t const_mask_width[1];  // mask width
__constant__ float const_mask_scale[1]; // mask scale
__constant__ int const_initialBit[1];
__constant__ int Filter[133]={
  0
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,1
  ,2
  ,3
  ,4
  ,5
  ,-1
  ,-1
  ,-1
  ,6
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,7
  ,8
  ,9
  ,10
  ,11
  ,12
  ,13
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,14
  ,15
  ,16
  ,17
  ,18
  ,-1
  ,19
  ,20
  ,21
  ,22
  ,23
  ,24
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,25
  ,26
  ,-1
  ,27
  ,-1
  ,-1
  ,-1
  ,-1
  ,28
  ,-1
  ,-1
  ,-1
  ,29
  ,30
  ,31
  ,-1
  ,-1
  ,-1
  ,-1
  ,32
  ,33
  ,-1
  ,34
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,-1
  ,35
  ,36
  ,37
  ,38
  ,39
  ,40
  ,41
  ,42
  ,-1
  ,-1
  ,43
  ,-1
  ,44
  ,45
  ,46
  ,-1
  ,-1
  ,-1
  ,-1
  ,47
  ,-1
  ,-1
  ,-1
  ,48
  ,49
};

// 
// __constant__ uint8_t const_numofcam[1];
// __constant__ gpu_voxels::Matrix4f const_intrInv[3]; // inverse intrinsic matrix
// __constant__ gpu_voxels::Matrix4f const_extrInv[3]; // inverse extrinsic matrix

template<class Voxel>
__global__
void kernelClearVoxelMap(Voxel* voxelmap, const uint32_t voxelmap_size)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    voxelmap[i] = Voxel();
  }
}

template<std::size_t bit_length>
__global__
void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, const uint32_t voxelmap_size, const uint32_t bit_index)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    BitVector<bit_length>& bit_vector = voxelmap[i].bitVector();
    if (bit_vector.getBit(bit_index))
      bit_vector.clearBit(bit_index);
  }
}

template<std::size_t bit_length>
__global__
void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, uint32_t voxelmap_size,
                         BitVector<bit_length> bits)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    BitVector<bit_length>& bit_vector = voxelmap[i].bitVector();
    if ((!(bit_vector & bits).isZero()))
    {
      BitVector<bit_length> tmp = bit_vector;
      tmp = tmp & (~bits);
      bit_vector = tmp;
    }
  }
}

/*! Collide two voxel maps.
 * Voxels are considered occupied for values
 * greater or equal given thresholds.
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMaps(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
                            Collider collider, bool* results)
{
  __shared__ bool cache[cMAX_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = false;
  bool temp = false;

  while (i < voxelmap_size)
  {
    if (collider.collide(voxelmap[i], other_map[i]))
    {
      temp = true;
    }
    i += blockDim.x * gridDim.x;
  }

  cache[cache_index] = temp;
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] || cache[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

// copy results from this block to global memory
  if (cache_index == 0)
  {
//    // FOR MEASUREMENT TEMPORARILY EDITED:
//    results[blockIdx.x] = true;
    results[blockIdx.x] = cache[0];
  }
}

/* Collide two voxel maps with storing collision info (for debugging only)
 * Voxels are considered occupied for values
 * greater or equal given thresholds.
 *
 * Collision info is stored within eBVM_COLLISION model for 'other_map'.
 * Warning: Original model is modified!
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMapsDebug(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
                                 Collider collider, uint16_t* results)
{
//#define DISABLE_STORING_OF_COLLISIONS
  __shared__ uint16_t cache[cMAX_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = 0;

  while (i < voxelmap_size)
  {
    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME meaning, static is used for debugging
    const bool collision = collider.collide(voxelmap[i], other_map[i]);
    if (collision) // store collision info
    {
#ifndef DISABLE_STORING_OF_COLLISIONS
//      other_map[i].occupancy = 255;
//      other_map[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
      voxelmap[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
#endif
      cache[cache_index] += 1;
    }
    i += blockDim.x * gridDim.x;
  }

  // debug: print collision coordinates

//  if (temp)
//  {
//    Vector3ui col_coord = mapToVoxels(voxelmap, dimensions, &(voxelmap[i]));
//    printf("Collision at voxel (%u) = (%u, %u, %u). Memory addresses are %p and %p.\n",
//           i, col_coord.x, col_coord.y, col_coord.z, (void*)&(voxelmap[i]), (void*)&(other_map[i]));
//  }
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] + cache[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    results[blockIdx.x] = cache[0];
  }
#undef DISABLE_STORING_OF_COLLISIONS
}


template<std::size_t length, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMapsBitvector(BitVoxel<length>* voxelmap, const uint32_t voxelmap_size,
                                     const OtherVoxel* other_map, Collider collider,
                                     BitVector<length>* results, uint16_t* num_collisions, const uint16_t sv_offset)
{
  extern __shared__ BitVector<length> cache[]; //[cMAX_THREADS_PER_BLOCK];
  __shared__ uint16_t cache_num[cMAX_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = BitVector<length>();
  cache_num[cache_index] = 0;
  BitVector<length> temp;

  while (i < voxelmap_size)
  {
    const bool collision = collider.collide(voxelmap[i], other_map[i], &temp, sv_offset);
    if (collision) // store collision info
    {
      #ifndef DISABLE_STORING_OF_COLLISIONS
      //      other_map[i].occupancy = 255;
      //      other_map[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
            voxelmap[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
      #endif
      cache[cache_index] = temp;
      cache_num[cache_index] += 1;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] | cache[cache_index + j];
      cache_num[cache_index] = cache_num[cache_index] + cache_num[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    // FOR MEASUREMENT TEMPORARILY EDITED:
    //results[blockIdx.x] = true;
    results[blockIdx.x] = cache[0];
    num_collisions[blockIdx.x] = cache_num[0];
  }
}

template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(Voxel* voxelmap, const Vector3ui dimensions, const float voxel_side_length,
                                  const Vector3f *points, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning,
                                  bool *points_outside_map)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_coords = mapToVoxels(voxel_side_length, points[i]);
    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y)
        && (uint_coords.z < dimensions.z))  // 해당 point cloud 가 map 범위에 존재하는 경우 //
    {
      // 해당 좌표를 index 로 변환한다음 //
      Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
      // eBVM_OCCUPIED : 1 의 값을 해당 index 에 넣어준다.
      voxel->insert(voxel_meaning);
    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//       printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
//              points[i].z);
    }
  }
}

//DistanceVoxel specialization
template<>
__global__
void kernelInsertGlobalPointCloud(DistanceVoxel* voxelmap, const Vector3ui dimensions, const float voxel_side_length,
                                  const Vector3f* points, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning,
                                  bool *points_outside_map)
{

  //debug
//  if (blockIdx.x + threadIdx.x == 0) {
//    printf("DEBUG: DistanceVoxelMap::insertPointCloud was called instead of the TemplateVoxelMap one\n");
//    for (uint32_t i = 0; i < sizePoints; i += 1) {
//      printf("DEBUG: point %u: %f %f %f\n", i, points[i].x, points[i].y, points[i].z);
//    }
//  }

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_coords = mapToVoxels(voxel_side_length, points[i]);
    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y)
        && (uint_coords.z < dimensions.z))
    {
      DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions,
                                                             uint_coords.x, uint_coords.y, uint_coords.z)];
      voxel->insert(uint_coords, voxel_meaning);
    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//      printf("DistanceVoxel kernelInsertGlobalPointCloud: Point (%u,%u,%u) is not in the range of the voxel map \n",
//             points[i].x, points[i].y, points[i].z);
    }
  }
}

// do added
__device__ __managed__ float recon_threshold = 0.05f;

__device__ __managed__ bool visUnknown = true;

__device__ __host__   __forceinline__
void update_threshold(float threshold)
{
  recon_threshold = threshold;
}

__device__ __host__   __forceinline__
void update_visUnknown(bool isVisUnknown)
{
  visUnknown = isVisUnknown;
}

__host__ __forceinline__
void setConstMem(const Vector3ui dims, const float voxel_side_length, const uint16_t depth_width, const uint16_t depth_height,
                  const uint16_t mask_width, const float scale)
{
  uint32_t map_dim[3] = {dims.x, dims.y, dims.z};
  float voxel_side[1] = {voxel_side_length};
  uint16_t depth_size[2] = {depth_width, depth_height};
  uint16_t mask_size[1] = {mask_width};
  float scale_size[1] = {scale};
  int initialBit[1] = {6};
  cudaMemcpyToSymbol(const_voxel_map_dim, &map_dim, sizeof(uint32_t)*3);
  cudaMemcpyToSymbol(const_voxel_side_length, &voxel_side, sizeof(float));
  cudaMemcpyToSymbol(const_depth_size, &depth_size, sizeof(uint16_t)*2);
  cudaMemcpyToSymbol(const_mask_width, &mask_size, sizeof(uint16_t));
  cudaMemcpyToSymbol(const_mask_scale, &scale_size, sizeof(float));
  cudaMemcpyToSymbol(const_initialBit, &initialBit, sizeof(int));
}

__global__
void kernelReconVoxelToDepthTriple(BitVectorVoxel* voxelmap, const Vector3ui dims, const float voxel_side_length, float* const depth0,float* const depth1,float* const depth2, uint8_t* const mask0,
                                uint8_t* const mask1, uint8_t* const mask2, gpu_voxels::Matrix4f const intrInv0, gpu_voxels::Matrix4f const intrInv1, gpu_voxels::Matrix4f const intrInv2, gpu_voxels::Matrix4f const extrInv0, 
                                gpu_voxels::Matrix4f const extrInv1, gpu_voxels::Matrix4f const extrInv2, const uint16_t depth_width,
                                const uint16_t depth_height, const uint16_t mask_width, const float scale)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  {
    bool isMasked=false;
    Vector3f uint_coords;
    uint_coords.x = (i % dims.x) * voxel_side_length;
    uint_coords.y = ((i / dims.x) % dims.y) * voxel_side_length;
    uint_coords.z = ((i / (dims.x*dims.y)) % dims.z) * voxel_side_length;
    BitVectorVoxel* voxel = &voxelmap[i];
    Vector4f proj;
    proj = ImageSpaceProjections(intrInv0, extrInv0, uint_coords);
    int x = (int) (proj.x / proj.z); int y = (int) (proj.y / proj.z);
    if(x>0 && y>0 && x<depth_width && y<depth_height)
    {
      float depthValue = depth0[y * depth_width + x];
      if(depthValue > 0) 
      {
        float dist = calculateDistCamtoVox(extrInv0, uint_coords);
        float t = fabsf(dist-depthValue);
        if( t < 0.05f) 
        {
          uint8_t maskid = mask0[static_cast<int>(round(scale * y)) * mask_width + static_cast<int>(round(scale *x))];
          for (int j = 10; j < 150; j++) {
            if(voxel->bitVector().getBit(j) == 1) {
              isMasked = true; // if voxel already has class data
            ;}
          }
          if(!isMasked){ //if voxel has no class data
            if(maskid == 0 && visUnknown) voxel->insert(static_cast<BitVoxelMeaning>(1)); // maskid 0 is unknown
            else if(maskid!=0 && voxel->bitVector().getBit(1)==1){ // if voxel has nonzero class data
              voxel->clear(static_cast<BitVoxelMeaning>(1));
              voxel->insert(static_cast<BitVoxelMeaning>(maskid-1+10));
            }
            else if(maskid!=0) voxel->insert(static_cast<BitVoxelMeaning>(maskid-1+10));
          }
        }
      }
    }
    if(!isMasked){
      proj = ImageSpaceProjections(intrInv1, extrInv1, uint_coords);
      x = (int) (proj.x / proj.z); y = (int) (proj.y / proj.z);
      if(x>0 && y>0 && x<depth_width && y<depth_height)
      {
        float depthValue = depth1[y * depth_width + x];
        if(depthValue > 0) 
        {
          float dist = calculateDistCamtoVox(extrInv1, uint_coords);
          float t = fabsf(dist-depthValue);
          if( t < 0.05f) 
          {
            uint8_t maskid = mask1[static_cast<int>(round(scale * y)) * mask_width + static_cast<int>(round(scale *x))];
            for (int j = 10; j < 150; j++) {
              if(voxel->bitVector().getBit(j) == 1) {
                isMasked = true; // if voxel already has class data
              ;}
            }
            if(!isMasked){ //if voxel has no class data
              if(maskid == 0 && visUnknown) voxel->insert(static_cast<BitVoxelMeaning>(1)); // maskid 0 is unknown
              else if(maskid!=0 && voxel->bitVector().getBit(1)==1){ // if voxel has nonzero class data
                voxel->clear(static_cast<BitVoxelMeaning>(1));
                voxel->insert(static_cast<BitVoxelMeaning>(maskid-1+10));
              }
              else if(maskid!=0) voxel->insert(static_cast<BitVoxelMeaning>(maskid-1+10));
            }
          }
        }
      }
    }

    if(!isMasked){
      proj = ImageSpaceProjections(intrInv2, extrInv2, uint_coords);
      x = (int) (proj.x / proj.z); y = (int) (proj.y / proj.z);
      if(x>0 && y>0 && x<depth_width && y<depth_height)
      {
        float depthValue = depth2[y * depth_width + x];
        if(depthValue > 0) 
        {
          float dist = calculateDistCamtoVox(extrInv2, uint_coords);
          float t = fabsf(dist-depthValue);
          if( t < 0.05f) 
          {
            uint8_t maskid = mask2[static_cast<int>(round(scale * y)) * mask_width + static_cast<int>(round(scale *x))];
            for (int j = 10; j < 150; j++) {
              if(voxel->bitVector().getBit(j) == 1) {
                isMasked = true; // if voxel already has class data
              ;}
            }
            if(!isMasked){ //if voxel has no class data
              if(maskid == 0 && visUnknown) voxel->insert(static_cast<BitVoxelMeaning>(1)); // maskid 0 is unknown
              else if(maskid!=0 && voxel->bitVector().getBit(1)==1){ // if voxel has nonzero class data
                voxel->clear(static_cast<BitVoxelMeaning>(1));
                voxel->insert(static_cast<BitVoxelMeaning>(maskid-1+10));
              }
              else if(maskid!=0) voxel->insert(static_cast<BitVoxelMeaning>(maskid-1+10));
            }
          }
        }
      }
    }
  }
}

__global__
void kernelReconVoxelToDepthTriple(BitVectorVoxel* voxelmap, const Vector3ui dims, const float voxel_side_length, float* const depth0,float* const depth1,float* const depth2,
                                gpu_voxels::Matrix4f const intrInv0, gpu_voxels::Matrix4f const intrInv1, gpu_voxels::Matrix4f const intrInv2, gpu_voxels::Matrix4f const extrInv0, 
                                gpu_voxels::Matrix4f const extrInv1, gpu_voxels::Matrix4f const extrInv2, const uint16_t depth_width,
                                const uint16_t depth_height)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  {
    Vector3f uint_coords;
    uint_coords.x = (i % dims.x) * voxel_side_length;
    uint_coords.y = ((i / dims.x) % dims.y) * voxel_side_length;
    uint_coords.z = ((i / (dims.x*dims.y)) % dims.z) * voxel_side_length;
    BitVectorVoxel* voxel = &voxelmap[i];
    Vector4f proj;
    proj = ImageSpaceProjections(intrInv0, extrInv0, uint_coords);
    int x = (int) (proj.x / proj.z); int y = (int) (proj.y / proj.z);
    if(x>0 && y>0 && x<depth_width && y<depth_height)
    {
      float depthValue = depth0[y * depth_width + x];
      if(depthValue > 0) 
      {
        float dist = calculateDistCamtoVox(extrInv0, uint_coords);
        float t = fabsf(dist-depthValue);
        if( t < 0.05f) 
        {
          voxel->insert(static_cast<BitVoxelMeaning>(1));return;
        }
      }
    }
    proj = ImageSpaceProjections(intrInv1, extrInv1, uint_coords);
    x = (int) (proj.x / proj.z); y = (int) (proj.y / proj.z);
    if(x>0 && y>0 && x<depth_width && y<depth_height)
    {
      float depthValue = depth1[y * depth_width + x];
      if(depthValue > 0) 
      {
        float dist = calculateDistCamtoVox(extrInv1, uint_coords);
        float t = fabsf(dist-depthValue);
        if( t < 0.05f) 
        {
          voxel->insert(static_cast<BitVoxelMeaning>(1));return;
        }
      }
    }
    proj = ImageSpaceProjections(intrInv2, extrInv2, uint_coords);
    x = (int) (proj.x / proj.z); y = (int) (proj.y / proj.z);
    if(x>0 && y>0 && x<depth_width && y<depth_height)
    {
      float depthValue = depth2[y * depth_width + x];
      if(depthValue > 0) 
      {
        float dist = calculateDistCamtoVox(extrInv2, uint_coords);
        float t = fabsf(dist-depthValue);
        if( t < 0.05f) 
        {
          voxel->insert(static_cast<BitVoxelMeaning>(1));return;
        }
      }
    }
  }
}

// this for depth and mask reconstruction without gvl->clearmap
__global__
void kernelReconWithPreprocess(BitVectorVoxel* voxelmap, float* const depth0,float* const depth1,float* const depth2, uint8_t* const mask0,
                                uint8_t* const mask1, uint8_t* const mask2, gpu_voxels::Matrix4f const intrInv0, gpu_voxels::Matrix4f const intrInv1, gpu_voxels::Matrix4f const intrInv2, gpu_voxels::Matrix4f const extrInv0, 
                                gpu_voxels::Matrix4f const extrInv1, gpu_voxels::Matrix4f const extrInv2)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; // 32bit = 4byte
  Vector3f uint_coords; // 12byte
  Vector4f proj; // 16byte
  uint_coords.x = (i % const_voxel_map_dim[0]) * const_voxel_side_length[0];
  uint_coords.y = ((i / const_voxel_map_dim[0]) % const_voxel_map_dim[1]) * const_voxel_side_length[0];
  uint_coords.z = ((i / (const_voxel_map_dim[0]*const_voxel_map_dim[1])) % const_voxel_map_dim[2]) * const_voxel_side_length[0];
  BitVectorVoxel* voxel = &voxelmap[i]; //256bit = 32byte
  proj = ImageSpaceProjections(intrInv0, extrInv0, uint_coords);
  int x0 = (int) __fdividef(proj.x, proj.z); int y0 = (int) __fdividef(proj.y, proj.z); // 8*3byte
  proj = ImageSpaceProjections(intrInv1, extrInv1, uint_coords);
  int x1 = (int) __fdividef(proj.x, proj.z); int y1 = (int) __fdividef(proj.y, proj.z); 
  proj = ImageSpaceProjections(intrInv2, extrInv2, uint_coords);
  int x2 = (int) __fdividef(proj.x, proj.z); int y2 = (int) __fdividef(proj.y, proj.z); 

  bool condition0 = (x0>0) & (y0>0) & (x0<const_depth_size[0]) & (y0<const_depth_size[1]);// 1*3byte
  bool condition1 = (x1>0) & (y1>0) & (x1<const_depth_size[0]) & (y1<const_depth_size[1]);
  bool condition2 = (x2>0) & (y2>0) & (x2<const_depth_size[0]) & (y2<const_depth_size[1]);

  bool isExist0 = chkVoxelinDepth(recon_threshold,extrInv0, uint_coords, depth0, const_depth_size[0], x0, y0, condition0);// 1*3byte
  bool isExist1 = chkVoxelinDepth(recon_threshold,extrInv1, uint_coords, depth1, const_depth_size[0], x1, y1, condition1);
  bool isExist2 = chkVoxelinDepth(recon_threshold,extrInv2, uint_coords, depth2, const_depth_size[0], x2, y2, condition2);
  bool isExist = isExist0 || isExist1 || isExist2;

  uint8_t maskid0 = isExist0 ? mask0[static_cast<int>(round(const_mask_scale[0] * y0)) * const_mask_width[0] + static_cast<int>(round(const_mask_scale[0] *x0))] : 0;//1*3byte
  uint8_t maskid1 = isExist1 ? mask1[static_cast<int>(round(const_mask_scale[0] * y1)) * const_mask_width[0] + static_cast<int>(round(const_mask_scale[0] *x1))] : 0;
  uint8_t maskid2 = isExist2 ? mask2[static_cast<int>(round(const_mask_scale[0] * y2)) * const_mask_width[0] + static_cast<int>(round(const_mask_scale[0] *x2))] : 0;

  // bool maskExist = (maskid0 != 0 || maskid1 != 0 || maskid2 != 0);
  bool maskExist = ((maskid0 | maskid1 | maskid2) != 0);
  int oldBit=voxel->bitVector().findBit();
  int maskinput=chkMask(oldBit, maskid0, maskid1, maskid2);
  int filtered = maskinput==-1?0: Filter[maskinput-1];
  uint8_t newBit = filtered==-1?0: filtered+const_initialBit[0];

   // not exist old voxel
  if(oldBit == -1) {
    if(isExist) {// new voxel exist
      if(maskExist){ // new voxel has mask data
        if(newBit!=0) voxel->insert(static_cast<BitVoxelMeaning>(newBit));
        return;}
      else{ // new voxel has no mask data
        if(visUnknown) voxel->insert(static_cast<BitVoxelMeaning>(1));
        return;}
    }
    else{ // new voxel not exist
      return;}
  }
  // exist old voxel
  else{
    if(isExist){ // new voxel exist
      if(maskExist){ // new voxel has mask data
        if(oldBit == newBit) return; // if state is not changed, do nothing
        else{ // if state is changed
          // printf("oldBit : %d , oldBit1 : %d\n", oldBit, oldBit1);
          voxel->clear(static_cast<BitVoxelMeaning>(oldBit));
          voxel->insert(static_cast<BitVoxelMeaning>(newBit));
          return;}
      }
      else{ // new voxel has no mask data
        if(!visUnknown) voxel->clear(static_cast<BitVoxelMeaning>(oldBit));
        return;} // just remmain old voxel's class data
    }
    else{ // new voxel not exist
      voxel->clear(static_cast<BitVoxelMeaning>(oldBit));return;}
  }
  // if(!visUnknown) voxel->clear(static_cast<BitVoxelMeaning>(1));
}


// this for only depth reconsturction without gvl->clearmap
__global__
void kernelReconWithPreprocess(BitVectorVoxel* voxelmap, float* const depth0,float* const depth1,float* const depth2, gpu_voxels::Matrix4f const intrInv0, 
                              gpu_voxels::Matrix4f const intrInv1, gpu_voxels::Matrix4f const intrInv2, gpu_voxels::Matrix4f const extrInv0, 
                              gpu_voxels::Matrix4f const extrInv1, gpu_voxels::Matrix4f const extrInv2)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  Vector3f uint_coords;
  uint_coords.x = (i % const_voxel_map_dim[0]) * const_voxel_side_length[0];
  uint_coords.y = ((i / const_voxel_map_dim[0]) % const_voxel_map_dim[1]) * const_voxel_side_length[0];
  uint_coords.z = ((i / (const_voxel_map_dim[0]*const_voxel_map_dim[1])) % const_voxel_map_dim[2]) * const_voxel_side_length[0];
  BitVectorVoxel* voxel = &voxelmap[i];
  bool oldBit = voxel->bitVector().getBit(1) == 1;
  // printf("oldBit : %d\n", oldBit);
  Vector4f proj;
  proj = ImageSpaceProjections(intrInv0, extrInv0, uint_coords);
  int x0 = (int) __fdividef(proj.x, proj.z); int y0 = (int) __fdividef(proj.y, proj.z);
  proj = ImageSpaceProjections(intrInv1, extrInv1, uint_coords);
  int x1 = (int) __fdividef(proj.x, proj.z); int y1 = (int) __fdividef(proj.y, proj.z);
  proj = ImageSpaceProjections(intrInv2, extrInv2, uint_coords);
  int x2 = (int) __fdividef(proj.x, proj.z); int y2 = (int) __fdividef(proj.y, proj.z);
  bool condition0 = (x0>0) & (y0>0) & (x0<const_depth_size[0]) & (y0<const_depth_size[1]);
  bool condition1 = (x1>0) & (y1>0) & (x1<const_depth_size[0]) & (y1<const_depth_size[1]);
  bool condition2 = (x2>0) & (y2>0) & (x2<const_depth_size[0]) & (y2<const_depth_size[1]);
  bool isExist = chkVoxelinDepth(recon_threshold, extrInv0, uint_coords, depth0,const_depth_size[0], x0, y0,condition0) || chkVoxelinDepth(recon_threshold, extrInv1, uint_coords, depth1, const_depth_size[0], x1, y1,condition1) || chkVoxelinDepth(recon_threshold, extrInv2, uint_coords, depth2, const_depth_size[0], x2, y2,condition2);

  // exist old voxel, not exist new voxel
  if(oldBit && !isExist) {
    voxel->clear(static_cast<BitVoxelMeaning>(1));return;
  }
  // not exist old voxel, exist new voxel
  else if(!oldBit && isExist) {
    voxel->insert(static_cast<BitVoxelMeaning>(1));return;
  }
  // if state is not changed, do nothing
  else 
    return;
}

// do added end

template<class Voxel>
__global__
void kernelInsertCoordinateTuples(Voxel* voxelmap, const Vector3ui dimensions, const float voxel_side_length,
                                  const Vector3ui *coordinates, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning,
                                  bool *points_outside_map)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_coords = coordinates[i];
    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y)
        && (uint_coords.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
      voxel->insert(voxel_meaning);
    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//       printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
//              points[i].z);
    }
  }
}

// DistanceVoxel specialization
template<>
__global__
void kernelInsertCoordinateTuples(DistanceVoxel* voxelmap, const Vector3ui dimensions, const float voxel_side_length,
                                  const Vector3ui *coordinates, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning,
                                  bool *points_outside_map)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_coords = coordinates[i];
    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y)
        && (uint_coords.z < dimensions.z))
    {
      DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
      voxel->insert(uint_coords, voxel_meaning);
    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//       printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
//              points[i].z);getVoxelIndexUnsigned
    }
  }
}

template<class Voxel>
__global__
void kernelInsertDilatedCoordinateTuples(Voxel* voxelmap, const Vector3ui dimensions,
                                  const Vector3ui *coordinates, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning,
                                  bool *points_outside_map)
{
  const int32_t SE_SIZE = 1;

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_center_coords = coordinates[i];
    // Check if center voxel is in range of the voxel map
    if ((uint_center_coords.x >= dimensions.x) || (uint_center_coords.y >= dimensions.y) || (uint_center_coords.z >= dimensions.z))
    {
      if(points_outside_map) *points_outside_map = true;
      continue;
    }

    // Iterate neighbors
    for (int32_t x = -SE_SIZE; x <= SE_SIZE; x++)
    {
      for (int32_t y = -SE_SIZE; y <= SE_SIZE; y++)
      {
        for (int32_t z = -SE_SIZE; z <= SE_SIZE; z++)
        {
          Vector3i int_neighbor_coords = Vector3i(uint_center_coords) + Vector3i(x, y, z);
          // Check if neighbor voxel is in range of the voxel map
          if ((int_neighbor_coords.x < dimensions.x) && (int_neighbor_coords.y < dimensions.y) && (int_neighbor_coords.z < dimensions.z)
              && (int_neighbor_coords.x >= 0) && (int_neighbor_coords.y >= 0) && (int_neighbor_coords.z >= 0))
          {
            Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, Vector3ui((uint32_t) int_neighbor_coords.x, (uint32_t) int_neighbor_coords.y, (uint32_t) int_neighbor_coords.z))];
            voxel->insert(voxel_meaning);
          }
        }
      }
    }
  }
}

template<class Voxel>
__global__
void kernelErode(Voxel* voxelmap_out, const Voxel* voxelmap_in, const Vector3ui dimensions, float erode_threshold, float occupied_threshold)
{
  const int32_t SE_SIZE = 1;
  Vector3ui uint_center_coords(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
  if ((uint_center_coords.x >= dimensions.x) || (uint_center_coords.y >= dimensions.y) || (uint_center_coords.z >= dimensions.z))
    return;

  // Count number of occupied neighbors, and total number of neighbors (might be less that 27 at map borders)
  uint32_t total = 0;
  uint32_t occupied = 0;
  for (int32_t x = -SE_SIZE; x <= SE_SIZE; x++)
  {
    for (int32_t y = -SE_SIZE; y <= SE_SIZE; y++)
    {
      for (int32_t z = -SE_SIZE; z <= SE_SIZE; z++)
      {
        const Vector3i int_neighbor_coords = Vector3i(uint_center_coords) + Vector3i(x, y, z);
        // Check if neighbor voxel is in range of the voxel map, and is not the center voxel
        if ((int_neighbor_coords.x < dimensions.x) && (int_neighbor_coords.y < dimensions.y) && (int_neighbor_coords.z < dimensions.z)
            && (int_neighbor_coords.x >= 0) && (int_neighbor_coords.y >= 0) && (int_neighbor_coords.z >= 0)
            && (x != 0 || y != 0 || z != 0))
        {
          total++;
          const Voxel& neighbor_voxel = voxelmap_in[getVoxelIndexUnsigned(dimensions, Vector3ui((uint32_t) int_neighbor_coords.x, (uint32_t) int_neighbor_coords.y, (uint32_t) int_neighbor_coords.z))];
          if (neighbor_voxel.isOccupied(occupied_threshold))
          {
            occupied++;
          }
        }
      }
    }
  }

  Voxel& voxel_out = voxelmap_out[getVoxelIndexUnsigned(dimensions, uint_center_coords)];
  if (((float) occupied) / total < erode_threshold)
  {
    // Clear voxel
    voxel_out = Voxel();
  }
  else
  {
    // Keep voxel
    voxel_out = voxelmap_in[getVoxelIndexUnsigned(dimensions, uint_center_coords)];
  }
}

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                BitVoxelMeaning voxel_meaning, const Vector3ui dimensions, const float voxel_side_length,
                                bool *points_outside_map)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_coords = mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y)
        && (uint_coords.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
      voxel->insert(voxel_meaning);

//        printf("Inserted Point @(%u,%u,%u) into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z);

    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//       printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
//              meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
//              meta_point_cloud->clouds_base_addresses[0][i].z);
    }
  }
}

//DistanceVoxel specialization
template<>
__global__
void kernelInsertMetaPointCloud(DistanceVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                BitVoxelMeaning voxel_meaning, const Vector3ui dimensions, const float voxel_side_length,
                                bool *points_outside_map)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    const Vector3ui uint_coordinates = mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((uint_coordinates.x < dimensions.x) && (uint_coordinates.y < dimensions.y)
        && (uint_coordinates.z < dimensions.z))
    {
      DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions,
                                                             uint_coordinates.x, uint_coordinates.y, uint_coordinates.z)];
      voxel->insert(uint_coordinates, voxel_meaning);

//        printf("Inserted Point @(%u,%u,%u) into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z);

    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//      printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
//             meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
//             meta_point_cloud->clouds_base_addresses[0][i].z);
    }
  }
}

//TODO: specialize every occurence of voxel->insert(meaning) for DistanceVoxel to use voxel->insert(integer_coordinates, meaning)

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                BitVoxelMeaning* voxel_meanings, const Vector3ui dimensions,
                                const float voxel_side_length,
                                bool *points_outside_map)
{
  u_int16_t sub_cloud = 0;
  u_int32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    // find out, to which sub_cloud our point belongs
    while(i >= sub_cloud_upper_bound)
    {
      sub_cloud++;
      sub_cloud_upper_bound += meta_point_cloud->cloud_sizes[sub_cloud];
    }


    const Vector3ui uint_coords = mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y)
        && (uint_coords.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
      voxel->insert(voxel_meanings[sub_cloud]);

//        printf("Inserted Point @(%u,%u,%u) with meaning %u into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z,
//               voxel_meanings[voxel_meaning_index]);

    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//       printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
//              meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
//              meta_point_cloud->clouds_base_addresses[0][i].z);
    }
  }
}


//BitVectorVoxel specialization
// This kernel may not be called with more threads than point per subcloud, as otherwise we will miss selfcollisions!
template<>
__global__
void kernelInsertMetaPointCloudSelfCollCheck(BitVectorVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                const BitVoxelMeaning* voxel_meanings, const Vector3ui dimensions, unsigned int sub_cloud,
                                const float voxel_side_length, const BitVector<BIT_VECTOR_LENGTH>* coll_masks,
                                bool *points_outside_map, BitVector<BIT_VECTOR_LENGTH>* colliding_subclouds)
{
  BitVector<BIT_VECTOR_LENGTH> masked;

  u_int32_t sub_cloud_upper_bound;
  Vector3ui uint_coords;

  sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sub_cloud_upper_bound;
      i += blockDim.x * gridDim.x)
  {
    uint_coords = mapToVoxels(voxel_side_length, meta_point_cloud->clouds_base_addresses[sub_cloud][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((uint_coords.x < dimensions.x) && (uint_coords.y < dimensions.y) && (uint_coords.z < dimensions.z))
    {
      BitVectorVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions, uint_coords)];
      masked.clear();
      masked = voxel->bitVector() & coll_masks[sub_cloud];
      if(! masked.noneButEmpty())
      {
        *colliding_subclouds |= masked; // copy the meanings of the colliding voxel, except the masked ones
        colliding_subclouds->setBit(voxel_meanings[sub_cloud]); // also set collisions for own meaning
        voxel->insert(eBVM_COLLISION); // Mark voxel as colliding
      }

      voxel->insert(voxel_meanings[sub_cloud]); // insert subclouds point


//        printf("Inserted Point @(%u,%u,%u) with meaning %u into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z,
//               voxel_meanings[voxel_meaning_index]);

    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
//       printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
//              meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
//              meta_point_cloud->clouds_base_addresses[0][i].z);
    }

  } // grid stride loop
//    unsigned int foo = atomicInc(global_sub_cloud_control, *global_sub_cloud_control);
//    printf("This thread inserted point %d, which was last of subcloud. Incrementing global control value to %d ...\n", i, (foo+1));
}




//DistanceVoxel specialization
template<>
__global__
void kernelInsertMetaPointCloud(DistanceVoxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                BitVoxelMeaning* voxel_meanings, const Vector3ui dimensions,
                                const float voxel_side_length, bool *points_outside_map)
{
  u_int16_t sub_cloud = 0;
  u_int32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    // find out, to which sub_cloud our point belongs
    while(i >= sub_cloud_upper_bound)
    {
      sub_cloud++;
      sub_cloud_upper_bound += meta_point_cloud->cloud_sizes[sub_cloud];
    }


    const Vector3ui uint_coordinates = mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((uint_coordinates.x < dimensions.x) && (uint_coordinates.y < dimensions.y)
        && (uint_coordinates.z < dimensions.z))
    {
      DistanceVoxel* voxel = &voxelmap[getVoxelIndexUnsigned(dimensions,
                                                             uint_coordinates.x, uint_coordinates.y, uint_coordinates.z)];
      voxel->insert(uint_coordinates, voxel_meanings[sub_cloud]);

//        printf("Inserted Point @(%u,%u,%u) with meaning %u into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z,
//               voxel_meanings[voxel_meaning_index]);

    }
    else
    {
      if(points_outside_map) *points_outside_map = true;
      /* printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
             meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
             meta_point_cloud->clouds_base_addresses[0][i].z); */
    }
  }
}

//template<std::size_t length>
//__global__
//void kernelInsertSensorDataWithRayCasting(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
//                                          const Vector3ui dimensions, const float voxel_side_length,
//                                          Sensor* sensor, const Vector3f* sensor_data,
//                                          const bool cut_real_robot, BitVoxel<length>* robotmap,
//                                          const uint32_t bit_index)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < sensor->data_size);
//      i += gridDim.x * blockDim.x)
//  {
//    if (!(isnan(sensor_data[i].x) || isnan(sensor_data[i].y) || isnan(sensor_data[i].z)))
//    {
//      const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
//      const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor->position);
//
//      /* both data and sensor coordinates must
//       be within boundaries for raycasting to work */
//      if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
//          && (integer_coordinates.z < dimensions.z) && (sensor_coordinates.x < dimensions.x)
//          && (sensor_coordinates.y < dimensions.y) && (sensor_coordinates.z < dimensions.z))
//      {
//        bool update = false;
//        if (robotmap && cut_real_robot)
//        {
//          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
//                                                      integer_coordinates.y, integer_coordinates.z);
//
////          if (!((robot_voxel->occupancy > 0) && (robot_voxel->voxelmeaning == eBVM_OCCUPIED))) // not occupied by robot
////           {
//          update = !robot_voxel->bitVector().getBit(bit_index); // not occupied by robot
////          else // else: sensor sees robot, no need to insert data.
////          {
////            printf("cutting robot from sensor data in kernel %u\n", i);
////          }
//        }
//        else
//          update = true;
//
//        if (update)
//        {
//          // sensor does not see robot, so insert data into voxelmap
//          // raycasting
//          rayCast(voxelmap, dimensions, sensor, sensor_coordinates, integer_coordinates);
//
//          // insert measured data itself:
//          ProbabilisticVoxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x,
//                                                  integer_coordinates.y, integer_coordinates.z);
//          voxel->updateOccupancy(cSENSOR_MODEL_OCCUPIED);
////            voxel->voxelmeaning = eBVM_OCCUPIED;
////            increaseOccupancy(voxel, cSENSOR_MODEL_OCCUPIED); // todo: replace with "occupied" of sensor model
//        }
//      }
//    }
//  }
//}

/* Insert sensor data into voxel map.
 * Assumes sensor data is already transformed
 * into world coordinate system.
 * If cut_real_robot is enabled one has to
 * specify pointer to the robot voxel map.
 * The robot voxels will be assumed 100% certain
 * and cut from sensor data.
 * See also function with ray casting.
 */
template<std::size_t length, class RayCasting>
__global__
void kernelInsertSensorData(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
                            const Vector3ui dimensions, const float voxel_side_length, const Vector3f sensor_pose,
                            const Vector3f* sensor_data, const size_t num_points, const bool cut_real_robot,
                            BitVoxel<length>* robotmap, const uint32_t bit_index, RayCasting rayCaster)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < num_points);
      i += gridDim.x * blockDim.x)
  {
    if (!(isnan(sensor_data[i].x) || isnan(sensor_data[i].y) || isnan(sensor_data[i].z)))
    {
      const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
      const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor_pose);

      /* both data and sensor coordinates must
       be within boundaries for raycasting to work */
      if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
          && (integer_coordinates.z < dimensions.z) && (sensor_coordinates.x < dimensions.x)
          && (sensor_coordinates.y < dimensions.y) && (sensor_coordinates.z < dimensions.z))
      {
        bool update = false;
        if (cut_real_robot)
        {
          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
                                                      integer_coordinates.y, integer_coordinates.z);

          update = !robot_voxel->bitVector().getBit(bit_index); // not occupied by robot
        }
        else
          update = true;

        if (update)
        {
          // sensor does not see robot, so insert data into voxelmap
          // raycasting
          rayCaster.rayCast(voxelmap, dimensions, sensor_coordinates, integer_coordinates);

          // insert measured data itself afterwards, so it overrides free voxels from raycaster:
          ProbabilisticVoxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x,
                                                  integer_coordinates.y, integer_coordinates.z);
          voxel->updateOccupancy(cSENSOR_MODEL_OCCUPIED);
        }
      }
    }
  }
}

template<std::size_t length>
__global__
void kernelShiftBitVector(BitVoxel<length>* voxelmap,
                          const uint32_t voxelmap_size, uint8_t shift_size)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    performLeftShift(voxelmap[i].bitVector(), shift_size);
  }
}

/**
 * cjuelg: jump flood distances, obstacle vectors
 *
 *
 * algorithm:
 *  calcNearestObstaclesJFA(VoxelMap, dim3, uint step_num (log(maxdim)..1))
 *       set map[x,y,z]= min(pos+{-1,0,1}*{x,y,z})
 */
__global__
void kernelJumpFlood3D(const DistanceVoxel * __restrict__ const voxels_input, DistanceVoxel* __restrict__ const voxels_output, const Vector3ui dims, const int32_t step_width){
  const uint32_t numVoxels = dims.x*dims.y*dims.z;

  //get linear address i
  //repeat if grid.x*block.x < numVoxels
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numVoxels; i += blockDim.x * gridDim.x)
  {
    //map to x,y,z
    Vector3i pos;
    pos.x = i % dims.x;
    pos.y = (i / dims.x) % dims.y;
    pos.z = (i / (dims.x*dims.y)) % dims.z;

    //check if point is in the range of the voxel map
    if ((pos.x < dims.x) && (pos.y < dims.y) && (pos.z < dims.z)) // should always be true
    {      
      DistanceVoxel min_voxel = voxels_input[i];

      if (min_voxel.squaredObstacleDistance(pos) == PBA_OBSTACLE_DISTANCE) {
        voxels_output[i] = min_voxel; // copy to buffer
        continue; //no other obstacle can be closer
      }

      // load 26 "step-neighbors"; for each: if distance is smaller, save obstacle and distance; (reduction operation)
      for (int x_step = -step_width; x_step <= step_width; x_step += step_width) {

        const int x_check = pos.x + x_step;
        if (x_check >= 0 && x_check < dims.x) { //don't leave map limits

          for (int y_step = -step_width; y_step <= step_width; y_step += step_width) {

            const int y_check = pos.y + y_step;
            if (y_check >= 0 && y_check < dims.y) { //don't leave map limits

              for (int z_step = -step_width; z_step <= step_width; z_step += step_width) {

                const int z_check = pos.z + z_step;
                if (z_check >= 0 && z_check < dims.z) { //don't leave map limits

                  if ((x_step != 0) || (y_step != 0) || (z_step != 0)) { //don't compare center_voxel to self
                    updateMinVoxel(voxels_input[getVoxelIndexSigned(dims, x_check, y_check, z_check)], min_voxel, pos);
                  }
                }
              }
            }
          }
        }
      }

      voxels_output[i] = min_voxel; //always update output array, even if min_voxel = voxels_input[i]
    }
    else
    {
      printf("(%i,%i,%i) is not in the range of the voxel map; SHOULD BE IMPOSSIBLE \n", pos.x, pos.y, pos.z);
    }
  }
}

/**
 * cjuelg: brute force exact obstacle distances
 *
 * optimisation1: check against known obstacle list instead of all other voxels
 * optimisation2: use shared memory to prefetch chunks of the obstacle list in parallel
 * optimisation3: resolve bank conflicts by using threadIdx as offset?
 */
__global__
void kernelExactDistances3D(DistanceVoxel* voxels, const Vector3ui dims, const float voxel_side_length,
                            Vector3f* obstacles, const std::size_t num_obstacles){

  extern __shared__ int dynamic_shared_mem[];
  Vector3i* obstacle_cache = (Vector3i*)dynamic_shared_mem; //size: cMAX_THREADS_PER_BLOCK * sizeof(DistanceVoxel)

  const uint32_t num_voxels = dims.x*dims.y*dims.z;

  //get linear address i

  if (gridDim.x * gridDim.y * blockDim.x < num_voxels)
    printf("exactDifferences3D: Alert: grids and blocks don't span num_voxels!");


  uint32_t voxel_idx = ((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x + threadIdx.x;
  if (voxel_idx >= num_voxels) return;
  DistanceVoxel* pos_voxel = &voxels[voxel_idx];

  Vector3i pos;
  pos.x = voxel_idx % dims.x;
  pos.y = (voxel_idx / dims.x) % dims.y;
  pos.z = (voxel_idx / (dims.x*dims.y)) % dims.z;

  int32_t min_distance = pos_voxel->squaredObstacleDistance(pos);
  Vector3i min_obstacle;

  for (uint obstacle_prefetch_offset = 0; obstacle_prefetch_offset < num_obstacles; obstacle_prefetch_offset += blockDim.x) {
    uint obstacle_prefetch_idx = obstacle_prefetch_offset + threadIdx.x;

    // prefetch
    if (obstacle_prefetch_idx < num_obstacles) {
      const Vector3i obstacle = mapToVoxelsSigned(voxel_side_length, obstacles[obstacle_prefetch_idx]);
      obstacle_cache[threadIdx.x] = obstacle;
    }
    __syncthreads();

    // update closest obstacle

    //check if point is in the range of the voxel map
    if ((pos.x < dims.x) && (pos.y < dims.y) && (pos.z < dims.z)) //always true?
    {
      if (min_distance != PBA_OBSTACLE_DISTANCE) { //else no other obstacle can be closer

        //check for every obstacle whether it is the closest one to pos
        for (uint s_obstacle_idx = 0; (s_obstacle_idx < cMAX_THREADS_PER_BLOCK) && (obstacle_prefetch_offset + s_obstacle_idx < num_obstacles); s_obstacle_idx++) {

          //optimise: resolve bank conflicts by using threadIdx as offset?
          //TODO: test optimisation, might even be slower (test using large obstacle count; with low obstacle count the kernel runs <2ms
          //          int cache_size = min(blockDim.x, (uint)num_obstacles - obstacle_prefetch_offset);
          //          const Vector3i obstacle_pos = obstacle_cache[(s_obstacle_idx + threadIdx.x ) % cache_size];
          const Vector3i obstacle_pos = obstacle_cache[s_obstacle_idx];

          //TODO: could perform sanity check, but: expensive, explodes number of memory accesses
//            const DistanceVoxel* other_voxel = &voxels[getVoxelIndex(dims, obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)];
//            if (other_voxel->getDistance() != DISTANCE_OBSTACLE) {
//              printf("ERROR: exactDistances3D: (pos: %i,%i,%i) given obstacle coordinates do not contain obstacle: (%u,%u,%u), %d\n", pos.x, pos.y, pos.z, obstacle_pos.x, obstacle_pos.y, obstacle_pos.z, other_voxel->getDistance());
//            }
          //            if (other_voxel != center_voxel && other_voxel->getDistance() == DISTANCE_OBSTACLE) {

          if (obstacle_pos != pos) {
            int32_t other_distance;
            if (
                  (obstacle_pos.x == PBA_UNINITIALISED_COORD)
                  || (obstacle_pos.y == PBA_UNINITIALISED_COORD)
                  || (obstacle_pos.z == PBA_UNINITIALISED_COORD)
                  || (pos.x == PBA_UNINITIALISED_COORD)
                  || (pos.y == PBA_UNINITIALISED_COORD)
                  || (pos.z == PBA_UNINITIALISED_COORD)
               )
            {
              other_distance = MAX_OBSTACLE_DISTANCE;

            } else {  // never use PBA_UNINIT in calculations
              const int dx = pos.x - obstacle_pos.x, dy = pos.y - obstacle_pos.y, dz = pos.z - obstacle_pos.z;
              other_distance = (dx * dx) + (dy * dy) + (dz * dz); //squared distance

              if (other_distance < min_distance) { //need to update minimum
                min_distance = other_distance;
                min_obstacle = obstacle_pos;
              }
            }
          }
        }
      }
    } else {
      printf("(%i,%i,%i) is not in the range of the voxel map; SHOULD BE IMPOSSIBLE \n", pos.x, pos.y, pos.z);
    }
    __syncthreads();
  }

  if (min_distance < pos_voxel->squaredObstacleDistance(pos)) { //need to update pos_voxel
    pos_voxel->setObstacle(min_obstacle);
  }
}

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
