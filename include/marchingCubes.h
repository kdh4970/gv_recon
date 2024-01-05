#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#ifndef __linux__
#define __linux__
#endif

#include <map>
#include <unordered_map>
#include <vector>
#include <GL/freeglut.h>
#include <helper_functions.h>

#include <cuda_runtime.h>

#include <cuda_gl_interop.h>
#include <tuple>
// #include <cuda_gl_interop.h>
#include <thread>
#include <mutex>
#include <string>
#include "cuda/marchingCubes_kernel.h"
#include <boost/shared_ptr.hpp>
#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>

#include "FastQuadric.h"

/// for removing duplicated vertex
template <typename TT>
struct hash_tuple {
    size_t operator()(TT const& tt) const { return std::hash<TT>()(tt); }
};

namespace {

template <class T>
inline void hash_combine(std::size_t& hash_seed, T const& v) {
    hash_seed ^= std::hash<T>()(v) + 0x9e3779b9 + (hash_seed << 6) +
                 (hash_seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(hash_seed, tuple);
        hash_combine(hash_seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        hash_combine(hash_seed, std::get<0>(tuple));
    }
};

}  // unnamed namespace


template <typename... TT>
struct hash_tuple<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
        size_t hash_seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(hash_seed, tt);
        return hash_seed;
    }
};


using boost::dynamic_pointer_cast;
using boost::shared_ptr;

extern std::mutex mtx;
extern bool isMask;
extern uint3 gridSizeLog2;
extern uint3 gridSizeShift;
extern uint3 gridSize;
extern uint3 gridSizeMask;
extern float3 voxelSize;
extern uint numVoxels;
extern uint maxVerts;

extern uchar* d_voxelRaw;
extern std::map<std::string, unsigned char> mask_decode_map;
extern unsigned char offset;

namespace MarchingCubes
{

///////////////////////////////////////
// forward declarations of GL and MC //
///////////////////////////////////////
/**
 * @brief Get the Device Voxel Data Object
 * 
 * @param ptrbitVoxmap : shared pointer of voxel map what type is gpu_voxels::voxelmap::BitVectorVoxelMap
 * @param numVoxels : number of voxels
 */
void getVoxelData(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap, const int numVoxels);

void run_glut_with_marchingCubes(int argc, char** argv);

void renderIsosurface();

void cleanup();

void initMC(int argc, char **argv);

void runGraphicsTest(int argc, char **argv); 

void computeIsosurface();

void computeIsosurface(uchar* mc_input);

void createVBO(GLuint *vbo, unsigned int size);

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource);

void display();

void keyboard(unsigned char key, int /*x*/, int /*y*/);

void mouse(int button, int state, int x, int y);

void motion(int x, int y);

void reshape(int w, int h);

void mainMenu(int i);

void animation();

void timerEvent(int value);

bool initGL(int *argc, char **argv);

void initMenus();

void computeFPS();

void SegmentMCinput(std::string target_class, int target_index);

void RemoveDuplicatedVertices(std::vector<float3> &vertices, std::vector<int3> &triangles);

void RemoveUnreferencedVertices(std::vector<float3> &vertices, std::vector<int3> &triangles);

void SimplifyMesh(std::vector<float3> &vertices, std::vector<int3> &triangles);

void SimplifyMeshSeg(std::string target_class, std::vector<float3> &vertices, std::vector<int3> &triangles);

void WriteTxtFile(std::string target_class, double decimation_ratio=0.5);

void SimplifyMeshAndUpload(std::string target_class, size_t target_idx, std::vector<float3> &vertices, std::vector<int3> &triangles, double decimation_ratio=0.5);

GLuint compileASMShader(GLenum program_type, const char *code);

void testSave();

}
#endif // MARCHINGCUBES_H

