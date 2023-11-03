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

/*
  Marching cubes

  This sample extracts a geometric isosurface from a volume dataset using
  the marching cubes algorithm. It uses the scan (prefix sum) function from
  the Thrust library to perform stream compaction.  Similar techniques can
  be used for other problems that require a variable-sized output per
  thread.

  For more information on marching cubes see:
  http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
  http://en.wikipedia.org/wiki/Marching_cubes

  Volume data courtesy:
  http://www9.informatik.uni-erlangen.de/External/vollib/

  For more information on the Thrust library
  http://code.google.com/p/thrust/

  The algorithm consists of several stages:

  1. Execute "classifyVoxel" kernel
  This evaluates the volume at the corners of each voxel and computes the
  number of vertices each voxel will generate.
  It is executed using one thread per voxel.
  It writes two arrays - voxelOccupied and voxelVertices to global memory.
  voxelOccupied is a flag indicating if the voxel is non-empty.

  2. Scan "voxelOccupied" array (using Thrust scan)
  Read back the total number of occupied voxels from GPU to CPU.
  This is the sum of the last value of the exclusive scan and the last
  input value.

  3. Execute "compactVoxels" kernel
  This compacts the voxelOccupied array to get rid of empty voxels.
  This allows us to run the complex "generateTriangles" kernel on only
  the occupied voxels.

  4. Scan voxelVertices array
  This gives the start address for the vertex data for each voxel.
  We read back the total number of vertices generated from GPU to CPU.

  Note that by using a custom scan function we could combine the above two
  scan operations above into a single operation.

  5. Execute "generateTriangles" kernel
  This runs only on the occupied voxels.
  It looks up the field values again and generates the triangle data,
  using the results of the scan to write the output to the correct addresses.
  The marching cubes look-up tables are stored in 1D textures.

  6. Render geometry
  Using number of vertices from readback.
*/
#ifndef __MC_CU__
#define __MC_CU__
#include "marchingCubes.h"





void dumpFile(void *dData, int data_bytes, const char *file_name) {
  void *hData = malloc(data_bytes);
  checkCudaErrors(cudaMemcpy(hData, dData, data_bytes, cudaMemcpyDeviceToHost));
  sdkDumpBin(hData, data_bytes, file_name);
  free(hData);
}

template <class T>
void dumpBuffer(T *d_buffer, int nelements, int size_element) {
  uint bytes = nelements * size_element;
  T *h_buffer = (T *)malloc(bytes);
  checkCudaErrors(
      cudaMemcpy(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < nelements; i++) {
    printf("%d: %u\n", i, h_buffer[i]);
  }

  printf("\n");
  free(h_buffer);
}


void runAutoTest(int argc, char **argv) {
  findCudaDevice(argc, (const char **)argv);

  // Initialize CUDA buffers for Marching Cubes
  initMC(argc, argv);

  computeIsosurface();

  char *ref_file = NULL;
  getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);

  enum DUMP_TYPE { DUMP_POS = 0, DUMP_NORMAL, DUMP_VOXEL };
  int dump_option = getCmdLineArgumentInt(argc, (const char **)argv, "dump");

  bool bTestResult = true;

  switch (dump_option) {
    case DUMP_POS:
      dumpFile((void *)d_pos, sizeof(float4) * maxVerts,
              "marchCube_posArray.bin");
      bTestResult = sdkCompareBin2BinFloat(
          "marchCube_posArray.bin", "posArray.bin",
          maxVerts * sizeof(float) * 4, EPSILON, THRESHOLD, argv[0]);
      break;

    case DUMP_NORMAL:
      dumpFile((void *)d_normal, sizeof(float4) * maxVerts,
              "marchCube_normalArray.bin");
      bTestResult = sdkCompareBin2BinFloat(
          "marchCube_normalArray.bin", "normalArray.bin",
          maxVerts * sizeof(float) * 4, EPSILON, THRESHOLD, argv[0]);
      break;

    case DUMP_VOXEL:
      dumpFile((void *)d_compVoxelArray, sizeof(uint) * numVoxels,
              "marchCube_compVoxelArray.bin");
      bTestResult = sdkCompareBin2BinFloat(
          "marchCube_compVoxelArray.bin", "compVoxelArray.bin",
          numVoxels * sizeof(uint), EPSILON, THRESHOLD, argv[0]);
      break;

    default:
      printf("Invalid validation flag!\n");
      printf("-dump=0 <check position>\n");
      printf("-dump=1 <check normal>\n");
      printf("-dump=2 <check voxel>\n");
      exit(EXIT_SUCCESS);
  }

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
// initialize marching cubes
////////////////////////////////////////////////////////////////////////////////
void initMC(int argc, char **argv) {
  // parse command line arguments
  int n;
  if (checkCmdLineFlag(argc, (const char **)argv, "grid")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "grid");
    gridSizeLog2.x = gridSizeLog2.y = gridSizeLog2.z = n;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "gridx")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "gridx");
    gridSizeLog2.x = n;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "gridy")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "gridy");
    gridSizeLog2.y = n;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "gridz")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "gridz");
    gridSizeLog2.z = n;
  }

  char *filename;

  if (getCmdLineArgumentString(argc, (const char **)argv, "file", &filename)) {
    volumeFilename = filename;
  }

  gridSize =
      make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
  gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
  gridSizeShift =
      make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

  numVoxels = gridSize.x * gridSize.y * gridSize.z;
  voxelSize =
      make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
  maxVerts = gridSize.x * gridSize.y * 100;

  printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z,
        numVoxels);
  printf("max verts = %d\n", maxVerts);

#if SAMPLE_VOLUME
  // load volume data
  char *path = sdkFindFilePath(volumeFilename, argv[0]);

  if (path == NULL) {
    fprintf(stderr, "Error finding file '%s'\n", volumeFilename);

    exit(EXIT_FAILURE);
  }

  int size = gridSize.x * gridSize.y * gridSize.z * sizeof(uchar);
  uchar *volume = loadRawFile(path, size);
  checkCudaErrors(cudaMalloc((void **)&d_volume, size));
  checkCudaErrors(cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice));
  free(volume);

  createVolumeTexture(d_volume, size);
#endif

  if (g_bValidate) {
    cudaMalloc((void **)&(d_pos), maxVerts * sizeof(float) * 4);
    cudaMalloc((void **)&(d_normal), maxVerts * sizeof(float) * 4);
  } else {
    // create VBOs
    createVBO(&posVbo, maxVerts * sizeof(float) * 4);
    // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(posVbo) );
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_posvbo_resource, posVbo, cudaGraphicsMapFlagsWriteDiscard));

    createVBO(&normalVbo, maxVerts * sizeof(float) * 4);
    // DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(normalVbo));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_normalvbo_resource, normalVbo, cudaGraphicsMapFlagsWriteDiscard));
  }
  // allocate textures
  allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

  // allocate device memory
  unsigned int memSize = sizeof(uint) * numVoxels;
  checkCudaErrors(cudaMalloc((void **)&d_voxelVerts, memSize));
  checkCudaErrors(cudaMalloc((void **)&d_voxelVertsScan, memSize));
  checkCudaErrors(cudaMalloc((void **)&d_voxelOccupied, memSize));
  checkCudaErrors(cudaMalloc((void **)&d_voxelOccupiedScan, memSize));
  checkCudaErrors(cudaMalloc((void **)&d_compVoxelArray, memSize));
}

void cleanup() {
  if (g_bValidate) {
    cudaFree(d_pos);
    cudaFree(d_normal);
  } else {
    sdkDeleteTimer(&timer);

    deleteVBO(&posVbo, &cuda_posvbo_resource);
    deleteVBO(&normalVbo, &cuda_normalvbo_resource);
  }
  destroyAllTextureObjects();
  checkCudaErrors(cudaFree(d_edgeTable));
  checkCudaErrors(cudaFree(d_triTable));
  checkCudaErrors(cudaFree(d_numVertsTable));

  checkCudaErrors(cudaFree(d_voxelVerts));
  checkCudaErrors(cudaFree(d_voxelVertsScan));
  checkCudaErrors(cudaFree(d_voxelOccupied));
  checkCudaErrors(cudaFree(d_voxelOccupiedScan));
  checkCudaErrors(cudaFree(d_compVoxelArray));

  if (d_volume) {
    checkCudaErrors(cudaFree(d_volume));
  }
}


void runGraphicsTest(int argc, char **argv) {
  printf("MarchingCubes\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
    printf("[%s]\n", argv[0]);
    printf("   Does not explicitly support -device=n in OpenGL mode\n");
    printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
    printf(" > %s -device=n -file=<reference> -dump=<0/1/2>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  if (false == initGL(&argc, argv)) {
    return;
  }

  findCudaDevice(argc, (const char **)argv);

  // register callbacks
  // glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  initMenus();

  // Initialize CUDA buffers for Marching Cubes
  initMC(argc, argv);

  sdkCreateTimer(&timer);

  // start rendering mainloop
  glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void computeIsosurface() {
  int threads = 128;
  dim3 grid(numVoxels / threads, 1, 1);

  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535) {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }

  // calculate number of vertices need per voxel
  launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_volume,
                      gridSize, gridSizeShift, gridSizeMask, numVoxels,
                      voxelSize, isoValue);
#if DEBUG_BUFFERS
  printf("voxelVerts:\n");
  dumpBuffer(d_voxelVerts, numVoxels, sizeof(uint));
#endif

#if SKIP_EMPTY_VOXELS
  // scan voxel occupied array
  ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

#if DEBUG_BUFFERS
  printf("voxelOccupiedScan:\n");
  dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif

  // read back values to calculate total number of non-empty voxels
  // since we are using an exclusive scan, the total is the last value of
  // the scan result plus the last value in the input array
  {
    uint lastElement, lastScanElement;
    checkCudaErrors(cudaMemcpy((void *)&lastElement,
                              (void *)(d_voxelOccupied + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
                              (void *)(d_voxelOccupiedScan + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost));
    activeVoxels = lastElement + lastScanElement;
  }

  if (activeVoxels == 0) {
    // return if there are no full voxels
    totalVerts = 0;
    return;
  }

  // compact voxel index array
  launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied,
                      d_voxelOccupiedScan, numVoxels);
  getLastCudaError("compactVoxels failed");

#endif  // SKIP_EMPTY_VOXELS

  // scan voxel vertex count array
  ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

#if DEBUG_BUFFERS
  printf("voxelVertsScan:\n");
  dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

  // readback total number of vertices
  {
    uint lastElement, lastScanElement;
    checkCudaErrors(cudaMemcpy((void *)&lastElement,
                              (void *)(d_voxelVerts + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
                              (void *)(d_voxelVertsScan + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost));
    totalVerts = lastElement + lastScanElement;
  }

  // generate triangles, writing to vertex buffers
  if (!g_bValidate) {
    size_t num_bytes;
    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_pos,
    // posVbo));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_posvbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_pos, &num_bytes, cuda_posvbo_resource));

    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_normal,
    // normalVbo));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_normalvbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_normal, &num_bytes, cuda_normalvbo_resource));
  }

#if SKIP_EMPTY_VOXELS
  dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
#else
  dim3 grid2((int)ceil(numVoxels / (float)NTHREADS), 1, 1);
#endif

  while (grid2.x > 65535) {
    grid2.x /= 2;
    grid2.y *= 2;
  }

#if SAMPLE_VOLUME
  launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
                            d_voxelVertsScan, d_volume, gridSize, gridSizeShift,
                            gridSizeMask, voxelSize, isoValue, activeVoxels,
                            maxVerts);
#else
  launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
                          d_voxelVertsScan, gridSize, gridSizeShift,
                          gridSizeMask, voxelSize, isoValue, activeVoxels,
                          maxVerts);
#endif

  if (!g_bValidate) {
    // DEPRECATED:      checkCudaErrors(cudaGLUnmapBufferObject(normalVbo));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_normalvbo_resource, 0));
    // DEPRECATED:      checkCudaErrors(cudaGLUnmapBufferObject(posVbo));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_posvbo_resource, 0));
  }
}








////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
    case (27):
      cleanup();
      exit(EXIT_SUCCESS);

    case '=':
      isoValue += 0.01f;
      break;

    case '-':
      isoValue -= 0.01f;
      break;

    case '+':
      isoValue += 0.1f;
      break;

    case '_':
      isoValue -= 0.1f;
      break;

    case 'w':
      wireframe = !wireframe;
      break;

    case ' ':
      animate = !animate;
      break;

    case 'l':
      lighting = !lighting;
      break;

    case 'r':
      render = !render;
      break;

    case 'c':
      compute = !compute;
      break;
  }

  printf("isoValue = %f\n", isoValue);
  printf("voxels = %d\n", activeVoxels);
  printf("verts = %d\n", totalVerts);
  printf("occupancy: %d / %d = %.2f%%\n", activeVoxels, numVoxels,
        activeVoxels * 100.0f / (float)numVoxels);

  if (!compute) {
    computeIsosurface();
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1 << button;
  } else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }

  mouse_old_x = x;
  mouse_old_y = y;
}

void motion(int x, int y) {
  float dx = (float)(x - mouse_old_x);
  float dy = (float)(y - mouse_old_y);

  if (mouse_buttons == 1) {
    rotate.x += dy * 0.2f;
    rotate.y += dx * 0.2f;
  } else if (mouse_buttons == 2) {
    translate.x += dx * 0.01f;
    translate.y -= dy * 0.01f;
  } else if (mouse_buttons == 3) {
    translate.z += dy * 0.01f;
  }

  mouse_old_x = x;
  mouse_old_y = y;
  glutPostRedisplay();
}

void idle() {
  animation();
  glutPostRedisplay();
}

void reshape(int w, int h) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)w / (float)h, 0.1, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);
}

void mainMenu(int i) { keyboard((unsigned char)i, 0, 0); }

#endif