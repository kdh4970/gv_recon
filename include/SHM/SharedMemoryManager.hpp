// do - 2023.12.13
// The test code for data sharing between C++ process and Python process by using shared memory
#pragma once
#include <cstring>
#include <string>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/types.h>
#include <vector>
#include <unistd.h>
#include "mesh.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <helper_math.h>

class SharedMemoryManager
{
public:
  /**
 * @brief Construct a new Shared Memory Manager:: Shared Memory Manager object
 * 
 * @param key  The key of shared memory
 * @param size  The size of shared memory
 */
  SharedMemoryManager()
  {
    _target = "all";
  }

  ~SharedMemoryManager()
  {
    // Detach Shared Memory
    shmdt(_shmptr);
    // Delete Shared Memory
    shmctl(_shmid, IPC_RMID, NULL);
    // Delete Semaphore
    semctl(_semid, 0, IPC_RMID);
  }
  
  void init(key_t key, std::string &target)
  {
    _size = 2*1024*1024;
    _key = key;
    _target = target;
    
    // Create Shared Memory if
    _shmid = shmget(_key, _size, IPC_CREAT | 0666);
    if (_shmid < 0)
    {
      perror("Failed to get shared memory id");
      exit(1);
    }
    
    // Create Semaphore
    _semid = semget(_key, 1, IPC_CREAT | 0666);
    semctl(_semid, 0, SETVAL, 1);
    

    // Attach Shared Memory
    _shmptr = (char*)shmat(_shmid, NULL, 0);
    if (_shmptr == (void *)-1) {
      perror("Failed to attach shared memory id");
      exit(1);
    }
    printf("Shared memory and Semaphore created. KEY = %d\n", _key);
  }


  void SendMesh(std::vector<float3> &vertices, std::vector<int3> &triangles)
  {
    printf("[DEBUG] : enter SendMesh\n");
    // First, Binary Serizlize by using Google Protocol Buffer
    Mesh mesh;
    mesh.set_class_info(_target.c_str());
    for(int i = 0; i < vertices.size(); i++)
    {
      mesh.add_vertices(vertices[i].x);
      mesh.add_vertices(vertices[i].y);
      mesh.add_vertices(vertices[i].z);
    }
    for(int i = 0; i < triangles.size(); i++)
    {
      mesh.add_triangles(triangles[i].x);
      mesh.add_triangles(triangles[i].y);
      mesh.add_triangles(triangles[i].z);
    }
    // for (int i = 0; i < vertices.size(); i++)
    // {
    //   Vertex vertex;
    //   vertex.add_position(vertices[i].x);
    //   vertex.add_position(vertices[i].y);
    //   vertex.add_position(vertices[i].z);
    //   mesh.add_vertices()->CopyFrom(vertex);
    // }
    // for (int i = 0; i < triangles.size(); i++)
    // {
    //   Triangle triangle;
    //   triangle.add_vertex_indices(triangles[i].x);
    //   triangle.add_vertex_indices(triangles[i].y);
    //   triangle.add_vertex_indices(triangles[i].z);
    //   mesh.add_triangles()->CopyFrom(triangle);
    // }

    std::string serialized_mesh;
    mesh.SerializeToString(&serialized_mesh);

    // Second, Send the serialized data to shared memory
    WriteStrToSharedMemory(serialized_mesh);
    // Add the Data allocation code here
    printf("serialized data size : %ld\n", serialized_mesh.size());
  }

  void WriteStrToSharedMemory(std::string &Serialized_data)
  {
    // Lock Semaphore
    _sb = {0, -1, 0};
    if (semop(_semid, &_sb, 1) == -1)
    {
      perror("Failed to lock semaphore");
      exit(1);
    }

    std::string output_string;
    google::protobuf::io::ArrayOutputStream array_output(_shmptr, Serialized_data.size() + sizeof(int));
    google::protobuf::io::CodedOutputStream coded_output(&array_output);

    coded_output.WriteVarint32(Serialized_data.size());  // write the size of data first
    coded_output.WriteRaw(Serialized_data.data(), Serialized_data.size());  // write the data

    memcpy(_shmptr, output_string.data(), output_string.size());

    // Unlock Semaphore
    _sb.sem_op = 1;
    if(semop(_semid, &_sb, 1))
    {
      perror("Failed to unlock semaphore");
      exit(1);
    }
    
  }


private:
  struct sembuf _sb;
  std::string _target;
  key_t _key;
  size_t _size;
  int _shmid, _semid;
  char* _shmptr;
};






