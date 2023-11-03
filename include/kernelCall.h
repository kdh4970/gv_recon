
#include <omp.h>
#include <helper_cuda.h>    // includes for helper CUDA functions
#include <helper_math.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "sMatrix.h"

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <stdint.h>


class DeviceKernel
{

public:
    DeviceKernel(){}
	DeviceKernel(const int inputWidth, const int inputHeight, const int numOfCam, std::vector<float4> ks);
	DeviceKernel(bool node_isMask, const int inputWidth, const int inputHeight, const int numOfCam, 
				std::vector<float4> ks, std::vector<Eigen::Matrix4f> ExtrInv, 
				gpu_voxels::Matrix4f ExtrInvGVox[3]);
	~DeviceKernel();


	double toDeviceMemory(const std::vector<float *> inputDepths);
	void toDeviceMemoryYosoArray(float* const inputDepths[3], uint8_t* inputMasks[3], uint16_t mask_width, uint16_t mask_height);
	double fromDevicePointToHostMem(std::vector<Vector3f*>* outputPointCloud);
	Vector3f* returnDevicePointer(int index)
	{return DevicePointCloud[index];}
    void testdepthToPointcloud(std::vector<Vector3f*>* points, float* depth,
	const Matrix4 invK, const int depthWidth, const int depthHeight);
	//do added
	double ReconVoxelToDepthtest(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap);
	double ReconVoxelWithPreprocess(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap);
	void saveVoxelRaw(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap);

private:
	int DepthWidth;
	int DepthHeight;
	uint2 depthsize;
	int NumofCam;
	int CamType;
	bool isMask;
	float Scale;
	std::vector<float*> DeviceDepth;
	std::vector<float4> intrinsics;
	std::vector<Vector3f*> DevicePointCloud;
	std::vector<Eigen::Matrix4f> IntrinsicEigen;
	std::vector<Eigen::Matrix4f> ExtrinsicsInv;
	dim3 imageBlocks = dim3(32, 16);
	Eigen::Matrix4f axisT;
	Eigen::Matrix4f axisTInv;
	Eigen::Matrix4f* cam1TransDevice;

	//do added
	float* DeviceDepthArray[3];
	uint8_t* DeviceYosoArray[3];
	gpu_voxels::Matrix4f DeviceExtrInvGVoxArray[3];
	gpu_voxels::Matrix4f DeviceIntrInvGVoxArray[3];
	uint16_t YosoMaskHeight, YosoMaskWidth;
	uchar* d_voxelRaw;
};

