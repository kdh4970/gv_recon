
#include "kernels.h"
#include <iostream>
#include <chrono>

DeviceKernel::DeviceKernel(const int inputWidth, const int inputHeight, const int numOfCam, std::vector<float4> ks)
{
	depthsize.x=inputWidth;
	depthsize.y=inputHeight;
	DepthWidth=inputWidth;
	DepthHeight=inputHeight;
	NumofCam=numOfCam;
	intrinsics=ks;
    std::cout <<"DEVICE KERNEL++++++++++++" <<NumofCam<<std::endl;
	DevicePointCloud.resize(NumofCam);
	DeviceDepth.resize(NumofCam);
	#pragma omp parallel for
	  for(int i= 0; i< NumofCam; i++)
	  {
		 cudaMalloc(&(DeviceDepth[i]), sizeof(float)*DepthWidth * DepthHeight);
		  cudaMalloc(&(DevicePointCloud[i]), sizeof(Vector3f)*DepthWidth * DepthHeight);
		  IntrinsicEigen.push_back(getInverseCameraMatrixEigen(intrinsics[i]));

	  }
	  axisT << 1, 0, 0, 0,
				0, 0, -1, 0,
				0, 1, 0, 0,
				0, 0, 0, 1;
	  axisTInv = axisT.inverse();


}

// yoso array
DeviceKernel::DeviceKernel(bool node_isMask, const int inputWidth, const int inputHeight, const int numOfCam, 
	std::vector<float4> ks, std::vector<Eigen::Matrix4f> ExtrInv, 
	gpu_voxels::Matrix4f ExtrInvGVox[3])
{   
	std::cout << "DeviceKernel Constructor!" << std::endl;
	DepthWidth=inputWidth;
	DepthHeight=inputHeight;
	isMask=node_isMask;

	NumofCam=numOfCam;
	intrinsics=ks;
	ExtrinsicsInv = ExtrInv;
	
	#pragma omp parallel for
	
	for(int i= 0; i< NumofCam; i++)
	{
		DeviceExtrInvGVoxArray[i]=ExtrInvGVox[i];

		gpu_voxels::Matrix4f tempintr = getIntrinsicGVox(intrinsics[i]);
		DeviceIntrInvGVoxArray[i] = tempintr;
		
		
		cudaError_t err = cudaMalloc(&(DeviceDepthArray[i]), sizeof(float) * DepthWidth * DepthHeight);
		if (err != cudaSuccess) {
				std::cout << "cudaMalloc failed with error code " << err << std::endl;
		}
		if(isMask){
			err = cudaMalloc(&(DeviceYosoArray[i]), sizeof(uint8_t)*DepthWidth * DepthHeight);
			if (err != cudaSuccess) {
					std::cout << "cudaMalloc failed with error code " << err << std::endl;
			}
			// prinf device yoso array address

			printf("GPU memory address: %p\n", DeviceYosoArray[i]);
		}
	}
	

	// std::cout << "Allocated GPU memory(intr,extr) : " << 9 * sizeof(Eigen::Matrix4f) << std::endl;
	// std::cout << "Allocated GPU memory(pcl) : " << 3*DepthWidth*DepthHeight*sizeof(Vector3f) + 3*DepthWidth*DepthHeight*sizeof(uint8_t) << std::endl;
}

DeviceKernel::~DeviceKernel()
{
	//#pragma omp parallel for
	for(int i=0; i< NumofCam; i++)
	{
		cudaFree(DeviceDepth[i]);
		DeviceDepth[i] = nullptr;
		cudaFree(DevicePointCloud[i]);
		DevicePointCloud[i] = nullptr;
		if(isMask){
			cudaFree(DeviceYosoArray[i]);
			DeviceYosoArray[i] =nullptr;
		}
		cudaFree(DeviceDepthArray[i]);
		DeviceDepthArray[i]=nullptr;
	}
}

double DeviceKernel::toDeviceMemory(const std::vector<float *> inputDepths)
{

	std::chrono::system_clock::time_point timeToDeviceStart = std::chrono::system_clock::now();
	//#pragma omp parallel for
	for(int i=0; i<NumofCam; i++)
	{
		cudaMemset(DeviceDepth[i], 0,DepthWidth*DepthHeight*sizeof(float));
		cudaMemcpy(DeviceDepth[i], inputDepths[i], DepthWidth*DepthHeight*sizeof(float), cudaMemcpyHostToDevice);
		//testdepthToPointcloud(HostPointCloud[0], inputDepths[0], ks[0], DepthWidth[RGB_R1536p],DepthHeight[RGB_R1536p]);
	}
	std::chrono::system_clock::time_point timeToDeviceEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> timecount = timeToDeviceEnd - timeToDeviceStart;
	//std::cout << "toDeviceMemory Time" << timecount.count() << " seconds" << std::endl;

	return timecount.count();

}

void DeviceKernel::toDeviceMemoryYosoArray(float* const inputDepths[3], uint8_t* inputMasks[3], uint16_t mask_width, uint16_t mask_height)
{
	// std::chrono::system_clock::time_point timeToDeviceStart = std::chrono::system_clock::now();
	YosoMaskHeight = mask_height;
	YosoMaskWidth = mask_width;

    Scale = static_cast<float>(mask_width) / DepthWidth;
	#pragma omp parallel for
	for(int i=0; i<NumofCam; i++)
	{
		cudaMemset(DeviceDepthArray[i], 0,DepthWidth*DepthHeight*sizeof(float));
		cudaMemcpy(DeviceDepthArray[i], inputDepths[i], DepthWidth*DepthHeight*sizeof(float), cudaMemcpyHostToDevice);
		if(isMask){
			cudaMemset(DeviceYosoArray[i], 0,DepthWidth*DepthHeight*sizeof(uint8_t));
			cudaMemcpy(DeviceYosoArray[i], inputMasks[i], YosoMaskHeight*YosoMaskWidth*sizeof(uint8_t), cudaMemcpyHostToDevice);
			
		}
	}
	// std::cout << "Allocated GPU memory(depth,mask) : " << 3*DepthWidth*DepthHeight*sizeof(float) + 3*DepthWidth*DepthHeight*sizeof(uint8_t) << std::endl;
	// float* DeviceDepthPointer[3] {DeviceDepthArray[0], DeviceDepthArray[1], DeviceDepthArray[2]};
	cudaDeviceSynchronize();
}


double DeviceKernel::fromDevicePointToHostMem(std::vector<Vector3f*>* outputPointCloud)
{
	std::chrono::system_clock::time_point timetoHostStart = std::chrono::system_clock::now();

	// cudaEvent_t start, stop;

	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, 0);

	//#pragma omp parallel for //2ms 차이
	for(int i=0; i<NumofCam; i++)
	{
		//std::cout << "[DEBUG] Device Point To Host Copy Thread : " << omp_get_thread_num() << " => " << i << "-th Device Pointer" << std::endl;	
		//std::cout << i << "-th Device Pointer Copy to Host Memory" << std::endl;	
		cudaMemset((*outputPointCloud)[i], 0, DepthWidth*DepthHeight*sizeof(Vector3f));
		cudaMemcpy((*outputPointCloud)[i], DevicePointCloud[i], DepthWidth*DepthHeight*sizeof(Vector3f), cudaMemcpyDeviceToHost);
	}
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);

	// float elapsedTime;
	// cudaEventElapsedTime(&elapsedTime, start, stop);

	// std::cout << "[DEBUG] Device Global Memory Copy To Host Time : " << elapsedTime << " ms" << std::endl;

	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);

	std::chrono::system_clock::time_point timetoHostEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> timecount = timetoHostEnd - timetoHostStart;
	//std::cout << "[DEBUG] Device Global Memory Copy To Host Time : " << timecount.count() << " seconds" << std::endl;

	return timecount.count();
}

void DeviceKernel::testdepthToPointcloud(std::vector<Vector3f*>* points, float* depth,
	const Matrix4 invK, const int depthWidth, const int depthHeight)
{
	for(int y=0;y<depthHeight;y++)
	{

    for(int x=0;x<depthWidth;x++)
	{
    	float depthValue = depth[depthWidth * y  + x];
		float4 temp;

		  temp =  (invK * make_float4(x, y, 1.f, 1.0f/depthValue));
		  temp.z = depthValue;
		  //printf("depthValue %f\n",  depth[depthWidth * y  + x]);
		  //std::cout << "x" << temp.x << std::endl;
		  //std::cout << "y" <<temp.y << std::endl;
	      //std::cout <<  "depth : " << depth[depthWidth * y  + x] << std::endl;
		  (*points)[0][depthWidth * y  + x]= Vector3f(depthValue*temp.x, depthValue*temp.y, temp.z);
		  //(*points)[0][depthWidth * y  + x]= Vector3f(depthValue*temp.x, temp.z, -depthValue*temp.y +6);



	}
	}

}

double DeviceKernel::ReconVoxelToDepthtest(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap)
{
	std::chrono::system_clock::time_point ReconStart = std::chrono::system_clock::now();
	// printf("intrinv : %.3f %.3f %.3f %.3f\n",(*DeviceIntrInvGVoxArray[0]).a11,(*DeviceIntrInvGVoxArray[0]).a12,(*DeviceIntrInvGVoxArray[0]).a13,(*DeviceIntrInvGVoxArray[0]).a14);
	
	
	// for(int i = 0; i < NumofCam; i++){
	// 	ptrbitVoxmap->ReconVoxelToDepth(DeviceDepthArray[i],DeviceYosoArray[i],DeviceIntrInvGVoxArray[i],DeviceExtrInvGVoxArray[i],DepthWidth, DepthHeight,YosoMaskWidth,Scale);
	// }


	if(isMask){
		ptrbitVoxmap->ReconVoxelToDepthTriple(DeviceDepthArray[0],DeviceDepthArray[1],DeviceDepthArray[2],DeviceYosoArray[0],DeviceYosoArray[1],DeviceYosoArray[2],
			DeviceIntrInvGVoxArray[0], DeviceIntrInvGVoxArray[1], DeviceIntrInvGVoxArray[2], DeviceExtrInvGVoxArray[0], DeviceExtrInvGVoxArray[1], DeviceExtrInvGVoxArray[2],
			DepthWidth, DepthHeight,YosoMaskWidth,Scale);}
	else{
		ptrbitVoxmap->ReconVoxelToDepthTriple(DeviceDepthArray[0],DeviceDepthArray[1],DeviceDepthArray[2], DeviceIntrInvGVoxArray[0],
			DeviceIntrInvGVoxArray[1], DeviceIntrInvGVoxArray[2], DeviceExtrInvGVoxArray[0], DeviceExtrInvGVoxArray[1], DeviceExtrInvGVoxArray[2],
			DepthWidth, DepthHeight);}
	// cudaDeviceSynchronize();
	std::chrono::system_clock::time_point ReconEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> timecount = ReconEnd - ReconStart;
	// std::cout << "[DEBUG] : (chrono::system_clock) Reconstruction Time          : " << timecount.count() * 1000 << " ms" << std::endl;
	return timecount.count()*1000;

}

double DeviceKernel::ReconVoxelWithPreprocess(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap)
{
	std::chrono::system_clock::time_point ReconStart = std::chrono::system_clock::now();
	if(isMask){
		// printf("in kernel cu : call mask version\n");
		ptrbitVoxmap->ReconstructionWithPreprocess(DeviceDepthArray[0],DeviceDepthArray[1],DeviceDepthArray[2],DeviceYosoArray[0],DeviceYosoArray[1],DeviceYosoArray[2],
			DeviceIntrInvGVoxArray[0], DeviceIntrInvGVoxArray[1], DeviceIntrInvGVoxArray[2], DeviceExtrInvGVoxArray[0], DeviceExtrInvGVoxArray[1], DeviceExtrInvGVoxArray[2],
			DepthWidth, DepthHeight,YosoMaskWidth,Scale);
		}
	else{
		ptrbitVoxmap->ReconstructionWithPreprocess(DeviceDepthArray[0],DeviceDepthArray[1],DeviceDepthArray[2], DeviceIntrInvGVoxArray[0],
			DeviceIntrInvGVoxArray[1], DeviceIntrInvGVoxArray[2], DeviceExtrInvGVoxArray[0], DeviceExtrInvGVoxArray[1], DeviceExtrInvGVoxArray[2],
			DepthWidth, DepthHeight);}
	// cudaDeviceSynchronize();
	std::chrono::system_clock::time_point ReconEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> timecount = ReconEnd - ReconStart;
	return timecount.count()*1000;
}

void DeviceKErnel