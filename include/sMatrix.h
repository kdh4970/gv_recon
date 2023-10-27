#ifndef sMATRIX_H
#define sMATRIX_H

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
//#include <cutil_math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <gpu_voxels/GpuVoxels.h>

struct sMatrix4 {
	float4 data[4];
};
struct Matrix4 {
	float4 data[4];
	Matrix4() {
	}
	Matrix4(sMatrix4 * src) {
		this->data[0] = make_float4(src->data[0].x, src->data[0].y,
				src->data[0].z, src->data[0].w);
		this->data[1] = make_float4(src->data[1].x, src->data[1].y,
				src->data[1].z, src->data[1].w);
		this->data[2] = make_float4(src->data[2].x, src->data[2].y,
				src->data[2].z, src->data[2].w);
		this->data[3] = make_float4(src->data[3].x, src->data[3].y,
				src->data[3].z, src->data[3].w);
	}

	inline __host__  __device__ float3 get_translation() const {
		return make_float3(data[0].w, data[1].w, data[2].w);
	}
};

inline cv::Matx33f RPYToRotation2(float R, float P, float Y)
{

    cv::Matx33d rotationTemp= {
                              cos(P)*cos(R),       sin(Y)*sin(P)*cos(R)-cos(Y)*sin(R),  cos(Y)*sin(P)*cos(R)+sin(Y)*sin(R),
                              cos(P)*sin(R),       sin(Y)*sin(P)*sin(R)+cos(Y)*cos(R),  cos(Y)*sin(P)*sin(R)-sin(Y)*cos(R),
                              -sin(P)      ,              sin(Y)*cos(P),                        cos(Y)*cos(P)
                              };
	/*cv::Matx33f rotationTemp= {
                              cos(P)*cos(R),                          -cos(P)*sin(Y),                      sin(P),
                              cos(Y)*sin(R)*sin(Y)+cos(R)*sin(Y),     cos(R)*cos(Y)-sin(R)*sin(P)*sin(Y),  -cos(P)*sin(R),
                              sin(R)*sin(Y)-cos(R)*cos(Y)*sin(P), cos(Y)*sin(R)+cos(R)*sin(P)*sin(Y),      cos(R)*cos(P)
                              };*/

 /*cv::Matx33f rotationTemp= {//RYP
                              cos(Y)*cos(P),                         -sin(Y),  cos(Y)*sin(P),
                              sin(R)*sin(P)+cos(R)*cos(P)*sin(Y),    cos(R)*sin(Y), cos(R)*sin(Y)*sin(P)-cos(P)*sin(R),
                              cos(P)*sin(R)*sin(Y)-cos(R)*cos(Y),  cos(Y)*sin(R), cos(P)*cos(R)+sin(R)*sin(P)*sin(Y)
                              };

    */
  /*cv::Matx33f rotationTemp= {//YPR
                         cos(Y)*cos(P), cos(Y)*cos(P)*sin(R)-cos(R)*sin(Y),  sin(R)*sin(Y)+cos(R)*cos(Y)*sin(P),
                         cos(P)*sin(Y), cos(R)*cos(Y)+sin(R)*sin(P)*sin(Y),  cos(R)*sin(P)*sin(Y)-cos(Y)*sin(R),
                         -sin(P), cos(P)*sin(R),      cos(R)*cos(P)
                         };*/
 return rotationTemp;
}

/*bool isRotationMatrix(cv::Matx33f R)
{
    cv::Matx33f Rt;
    cv::transpose(R, Rt);
    cv::Matx33f shouldBeIdentity = Rt * R;
    cv::Matx33f I = I.eye();

    return  cv::norm(I, shouldBeIdentity) < 1e-6;

}*/

inline Matrix4 getInverseCameraMatrix(const float4 & k)
{
	Matrix4 invK;
	invK.data[0] = make_float4(1.0f / k.x, 0, -k.z / k.x, 0);
	invK.data[1] = make_float4(0, 1.0f / k.y, -k.w / k.y, 0);
	invK.data[2] = make_float4(0, 0, 1, 0);
	invK.data[3] = make_float4(0, 0, 0, 1);
	return invK;
}
inline Eigen::Matrix4f getInverseCameraMatrixEigen(const float4 & k)
{
	Eigen::Matrix4f invK;
	invK << 1.0f / k.x, 0, -k.z / k.x, 0,
			0, 1.0f / k.y, -k.w / k.y, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
	return invK;
}
inline gpu_voxels::Matrix4f getInverseCameraMatrixGVox(const float4 & k) 
{
	
	gpu_voxels::Matrix4f invK(1.0f / k.x, 0, -k.z / k.x, 0,
							  0, 1.0f / k.y, -k.w / k.y, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1);
	return invK;
}

inline Eigen::Matrix4f setExtrinsicEigen(float x, float y, float z, float roll, float pitch, float yaw)
{ //https://blog.naver.com/PostView.nhn?blogId=junghs1040&logNo=222104963167
	cv::Matx33d rot;
    rot = rot.eye();
    rot =RPYToRotation2(yaw, pitch, roll);
	//rot =RPYToRotation2(roll, pitch, yaw);
	//isRotationMatrix(rot);
	Eigen::Matrix4f extr;

	extr << rot(0,0), rot(0,1), rot(0,2) , x,
			rot(1,0), rot(1,1), rot(1,2), y,
			rot(2,0), rot(2,1), rot(2,2), z,
			0, 0, 0, 1;

	//printf("Eigen extr: %f, %f, %f\n",extrInv(0,0),extrInv(0,1), extrInv(0,2) );
	return extr;
}


inline gpu_voxels::Matrix4f getIntrinsicGVox(const float4 & k) 
{
	gpu_voxels::Matrix4f invK(k.x, 0, k.z, 0,
							  0, k.y, k.w, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1);
	return invK;
}

inline gpu_voxels::Matrix4f setExtrinsicGVox(float x, float y, float z, float roll, float pitch, float yaw)
{ 
	cv::Matx33d rot;
    rot = rot.eye();
    rot =RPYToRotation2(yaw, pitch, roll);
	//rot =RPYToRotation2(roll, pitch, yaw);
	//isRotationMatrix(rot);
	gpu_voxels::Matrix4f extr(rot(0,0), rot(0,1), rot(0,2) , x,
			       rot(1,0), rot(1,1), rot(1,2), y,
			       rot(2,0), rot(2,1), rot(2,2), z,
			       0, 0, 0, 1);
	
    //gpu_voxels::Matrix4f returntemp = extr * axis;
	gpu_voxels::Matrix4f returntemp = extr;
	
	//printf("extr returntemp : %f,  %f  %f, %f\n", returntemp.a11, returntemp.a12, returntemp.a13, returntemp.a14);	
	return returntemp;

}

inline Eigen::Matrix4f setExtrinsicInvEigen(float x, float y, float z, float roll, float pitch, float yaw)
{ //https://blog.naver.com/PostView.nhn?blogId=junghs1040&logNo=222104963167
	cv::Matx33d rot;
    rot = rot.eye();
    rot =RPYToRotation2(yaw, pitch, roll);
	//rot =RPYToRotation2(roll, pitch, yaw);
	//isRotationMatrix(rot);
	Eigen::Matrix4f extr;

	extr << rot(0,0), rot(0,1), rot(0,2) , x,
			rot(1,0), rot(1,1), rot(1,2), y,
			rot(2,0), rot(2,1), rot(2,2), z,
			0, 0, 0, 1;
	// printf("\nEigen extr:\n");		
	// printf("%f, %f, %f\n",extr(0,0),extr(0,1), extr(0,2) );
	// printf("%f, %f, %f\n",extr(1,0),extr(1,1), extr(1,2) );
	// printf("%f, %f, %f\n",extr(2,0),extr(2,1), extr(2,2) );
	
	Eigen::Matrix4f extrInv = extr.inverse();

	//printf("Eigen extr: %f, %f, %f\n",extrInv(0,0),extrInv(0,1), extrInv(0,2) );
	return extrInv;

}

inline Eigen::Matrix4f setExtrinsicInvEigen(float x, float y, float z, float roll, float pitch, float yaw, Eigen::Matrix4f cam1Inv)
{ //https://blog.naver.com/PostView.nhn?blogId=junghs1040&logNo=222104963167
	cv::Matx33d rot;
    rot = rot.eye();
    rot =RPYToRotation2(yaw, pitch, roll);
	Eigen::Matrix4f extr;

	extr << rot(0,0), rot(0,1), rot(0,2) , x,
			rot(1,0), rot(1,1), rot(1,2), y,
			rot(2,0), rot(2,1), rot(2,2), z,
			0, 0, 0, 1;
	// extr << 1, 0, 0 , 0,
	// 		0, 1, 0, 0,
	// 		0, 0, 1, 0,
	// 		0, 0, 0, 1;
	// printf("Eigen extr: %f, %f, %f\n", extr(0,0), extr(0,1), extr(0,2) );
	// Eigen::Matrix4f extrInv = extr.inverse()*cam1Inv;
	Eigen::Matrix4f extrInv = extr.inverse();
	// printf("Eigen extr: %f, %f, %f\n",extrInv(0,0),extrInv(0,1), extrInv(0,2) );
	return extrInv;

}



inline Matrix4 setExtrinsic(float x, float y, float z, float roll, float pitch, float yaw)
{
	cv::Matx33d rot;
    rot = rot.eye();
    rot =RPYToRotation2(yaw, pitch, roll);
	//rot =RPYToRotation2(roll, pitch, yaw);
	//isRotationMatrix(rot);
	Matrix4 extr;
	extr.data[0] = make_float4(rot(0,0), rot(0,1), rot(0,2) , x);
	extr.data[1] = make_float4(rot(1,0), rot(1,1), rot(1,2), y);
	extr.data[2] = make_float4(rot(2,0), rot(2,1), rot(2,2), z);
	extr.data[3] = make_float4(0, 0, 0, 1);
	return extr;

}
inline Matrix4 setExtrinsicInv(float x, float y, float z, float roll, float pitch, float yaw)
{
	cv::Matx33d rot;
    rot = rot.eye();
    rot =RPYToRotation2(yaw, pitch, roll);
	//rot =RPYToRotation2(roll, pitch, yaw);
	//isRotationMatrix(rot);
	Matrix4 extr;

	extr.data[0] = make_float4(rot(0,0), rot(1,0), rot(2,0) , 0);
	extr.data[1] = make_float4(rot(0,1), rot(1,1), rot(2,1), 0);
	extr.data[2] = make_float4(rot(0,2), rot(1,2), rot(2,2), 0);
	extr.data[3] = make_float4(x, y, z, 1);

	printf("setExtrInv: %f, %f, %f\n",extr.data[0].x,extr.data[0].y, extr.data[0].z );
	return extr;

}


inline __host__  __device__ float3 rotate(const Matrix4 & M, const float3 & v) {
	return make_float3(dot(make_float3(M.data[0]), v),
			dot(make_float3(M.data[1]), v), dot(make_float3(M.data[2]), v));
}

inline __host__  __device__ float4 operator*(const Matrix4 & M, const float4 & v) {
	return make_float4(dot(M.data[0], v), dot(M.data[1], v), dot(M.data[2], v),
			dot(M.data[3], v));
}


/*inline __host__  __device__ float4 operator*(float a, const float4 & v) {
	return make_float4( a * v.x, a* v.y, a* v.z, a* v.w);
}*/

inline int divup(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
inline dim3 divup(uint2 a, dim3 b) {
	return dim3(divup(a.x, b.x), divup(a.y, b.y));
}
inline dim3 divup(dim3 a, dim3 b) {
	return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z));
}

//dim3 imageBlocks = dim3(32, 16);
#endif
