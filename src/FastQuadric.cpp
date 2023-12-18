#include "FastQuadric.h"

FastQuadricDecimator::FastQuadricDecimator(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles)
{
	Vertex tempVertex;
	Triangle tempTriangle;
	
	// second. insert data
	for(auto i: vertices)
	{
		tempVertex.p.x = i(0);
		tempVertex.p.y = i(1);
		tempVertex.p.z = i(2);
		_vertices.push_back(tempVertex);
	}
	for(auto i: triangles)
	{
		// write something...
		tempTriangle.v[0] = i(0);
		tempTriangle.v[1] = i(1);
		tempTriangle.v[2] = i(2);

		_triangles.push_back(tempTriangle);
	}
}

FastQuadricDecimator::~FastQuadricDecimator()
{
}

size_t FastQuadricDecimator::getVertexCount() {return _vertices.size();}
size_t FastQuadricDecimator::getTriangleCount() {return _triangles.size();}

void FastQuadricDecimator::save_txt(std::string filename){
	  // Open target obj file to write
	FILE *fp=fopen(filename.c_str(), "w");
	if(!fp)
	{
		printf("Error: write obj file error.\n");
		return;
	}
	// Write vertex list
	for(int i=0;i<getVertexCount();i++)
	{
		Vertex &v=_vertices[i];
		fprintf(fp,"v %f %f %f\n",v.p.x,v.p.y,v.p.z);
	}
	// Write face list
	for(int i=0;i<getTriangleCount();i++)
	{
		Triangle &t=_triangles[i];
		fprintf(fp,"f %d %d %d\n",t.v[0],t.v[1],t.v[2]);
	}
	fclose(fp);
}

void FastQuadricDecimator::simplify_mesh(int target_count, double agressiveness=7, bool verbose=false)
	{
		// init
		loopi(0,_triangles.size())
        {
            _triangles[i].deleted=0;
        }
		// main iteration loop
		int deleted_triangles=0;
		std::vector<int> deleted0,deleted1;
		int triangle_count=_triangles.size();
		//int iteration = 0;
		//loop(iteration,0,100)
		for (int iteration = 0; iteration < 100; iteration ++)
		{
			if(triangle_count-deleted_triangles<=target_count)break;
			// update mesh once in a while
			if(iteration%5==0)
			{
				update_mesh(iteration);
			}
			// clear dirty flag
			loopi(0,_triangles.size()) _triangles[i].dirty=0;
			//
			// All triangles with edges below the threshold will be removed
			//
			// The following numbers works well for most models.
			// If it does not, try to adjust the 3 parameters
			//
			double threshold = 0.000000001*pow(double(iteration+3),agressiveness);

			// target number of triangles reached ? Then break
			if ((verbose) && (iteration%5==0)) {
				printf("| iteration %d - triangles %d threshold %g\n",iteration,triangle_count-deleted_triangles, threshold);
			}
			// remove vertices & mark deleted triangles
			loopi(0,_triangles.size())
			{
				Triangle &t=_triangles[i];
				if(t.err[3]>threshold) continue;
				if(t.deleted) continue;
				if(t.dirty) continue;

				loopj(0,3)if(t.err[j]<threshold)
				{
					int i0=t.v[ j     ]; Vertex &v0 = _vertices[i0];
					int i1=t.v[(j+1)%3]; Vertex &v1 = _vertices[i1];
					// Border check
					if(v0.border != v1.border)  continue;

					// Compute vertex to collapse to
					vec3f p;
					calculate_error(i0,i1,p);
					deleted0.resize(v0.tcount); // normals temporarily
					deleted1.resize(v1.tcount); // normals temporarily
					// don't remove if flipped
					if( flipped(p,i0,i1,v0,v1,deleted0) ) continue;

					if( flipped(p,i1,i0,v1,v0,deleted1) ) continue;

					
					// not flipped, so remove edge
					v0.p=p;
					v0.q=v1.q+v0.q;
					int tstart=_refs.size();

					update_triangles(i0,v0,deleted0,deleted_triangles);
					update_triangles(i0,v1,deleted1,deleted_triangles);

					int tcount=_refs.size()-tstart;
					if(tcount<=v0.tcount)
					{
						// save ram
						if(tcount)memcpy(&_refs[v0.tstart],&_refs[tstart],tcount*sizeof(Ref));
					}
					else
						// append
						v0.tstart=tstart;

					v0.tcount=tcount;
					break;
				}
				// done?
				if(triangle_count-deleted_triangles<=target_count)break;
			}
		}
		// clean up mesh
		compact_mesh();
	} //simplify_mesh()

// Check if a triangle flips when this edge is removed

bool FastQuadricDecimator::flipped(vec3f p,int i0,int i1,Vertex &v0,Vertex &v1,std::vector<int> &deleted)
{

	loopk(0,v0.tcount)
	{
		Triangle &t=_triangles[_refs[v0.tstart+k].tid];
		if(t.deleted)continue;

		int s=_refs[v0.tstart+k].tvertex;
		int id1=t.v[(s+1)%3];
		int id2=t.v[(s+2)%3];

		if(id1==i1 || id2==i1) // delete ?
		{

			deleted[k]=1;
			continue;
		}
		vec3f d1 = _vertices[id1].p-p; d1.normalize();
		vec3f d2 = _vertices[id2].p-p; d2.normalize();
		if(fabs(d1.dot(d2))>0.999) return true;
		vec3f n;
		n.cross(d1,d2);
		n.normalize();
		deleted[k]=0;
		if(n.dot(t.n)<0.2) return true;
	}
	return false;
}


// Update triangle connections and edge error after a edge is collapsed

void FastQuadricDecimator::update_triangles(int i0,Vertex &v,std::vector<int> &deleted,int &deleted_triangles)
{
	vec3f p;
	loopk(0,v.tcount)
	{
		Ref &r=_refs[v.tstart+k];
		Triangle &t=_triangles[r.tid];
		if(t.deleted)continue;
		if(deleted[k])
		{
			t.deleted=1;
			deleted_triangles++;
			continue;
		}
		t.v[r.tvertex]=i0;
		t.dirty=1;
		t.err[0]=calculate_error(t.v[0],t.v[1],p);
		t.err[1]=calculate_error(t.v[1],t.v[2],p);
		t.err[2]=calculate_error(t.v[2],t.v[0],p);
		t.err[3]=min(t.err[0],min(t.err[1],t.err[2]));
		_refs.push_back(r);
	}
}

// compact triangles, compute edge error and build reference list

void FastQuadricDecimator::update_mesh(int iteration)
{
	if(iteration>0) // compact triangles
	{
		int dst=0;
		loopi(0,_triangles.size())
		if(!_triangles[i].deleted)
		{
			_triangles[dst++]=_triangles[i];
		}
		_triangles.resize(dst);
	}

	// Init Reference ID list
	loopi(0,_vertices.size())
	{
		_vertices[i].tstart=0;
		_vertices[i].tcount=0;
	}
	loopi(0,_triangles.size())
	{
		Triangle &t=_triangles[i];
		loopj(0,3) _vertices[t.v[j]].tcount++;
	}
	int tstart=0;
	loopi(0,_vertices.size())
	{
		Vertex &v=_vertices[i];
		v.tstart=tstart;
		tstart+=v.tcount;
		v.tcount=0;
	}

	// Write References
	_refs.resize(_triangles.size()*3);
	loopi(0,_triangles.size())
	{
		Triangle &t=_triangles[i];
		loopj(0,3)
		{
			Vertex &v=_vertices[t.v[j]];
			_refs[v.tstart+v.tcount].tid=i;
			_refs[v.tstart+v.tcount].tvertex=j;
			v.tcount++;
		}
	}

	// Init Quadrics by Plane & Edge Errors
	//
	// required at the beginning ( iteration == 0 )
	// recomputing during the simplification is not required,
	// but mostly improves the result for closed meshes
	//
	if( iteration == 0 )
	{
		// Identify boundary : vertices[].border=0,1

		std::vector<int> vcount,vids;

		loopi(0,_vertices.size())
			_vertices[i].border=0;

		loopi(0,_vertices.size())
		{
			Vertex &v=_vertices[i];
			vcount.clear();
			vids.clear();
			loopj(0,v.tcount)
			{
				int k=_refs[v.tstart+j].tid;
				Triangle &t=_triangles[k];
				loopk(0,3)
				{
					int ofs=0,id=t.v[k];
					while(ofs<vcount.size())
					{
						if(vids[ofs]==id)break;
						ofs++;
					}
					if(ofs==vcount.size())
					{
						vcount.push_back(1);
						vids.push_back(id);
					}
					else
						vcount[ofs]++;
				}
			}
			loopj(0,vcount.size()) if(vcount[j]==1)
				_vertices[vids[j]].border=1;
		}
		//initialize errors
		loopi(0,_vertices.size())
			_vertices[i].q=SymetricMatrix(0.0);

		loopi(0,_triangles.size())
		{
			Triangle &t=_triangles[i];
			vec3f n,p[3];
			loopj(0,3) p[j]=_vertices[t.v[j]].p;
			n.cross(p[1]-p[0],p[2]-p[0]);
			n.normalize();
			t.n=n;
			loopj(0,3) _vertices[t.v[j]].q =
				_vertices[t.v[j]].q+SymetricMatrix(n.x,n.y,n.z,-n.dot(p[0]));
		}
		loopi(0,_triangles.size())
		{
			// Calc Edge Error
			Triangle &t=_triangles[i];vec3f p;
			loopj(0,3) t.err[j]=calculate_error(t.v[j],t.v[(j+1)%3],p);
			t.err[3]=min(t.err[0],min(t.err[1],t.err[2]));
		}
	}
}

// Finally compact mesh before exiting

void FastQuadricDecimator::compact_mesh()
{
	int dst=0;
	loopi(0,_vertices.size())
	{
		_vertices[i].tcount=0;
	}
	loopi(0,_triangles.size())
	if(!_triangles[i].deleted)
	{
		Triangle &t=_triangles[i];
		_triangles[dst++]=t;
		loopj(0,3)_vertices[t.v[j]].tcount=1;
	}
	_triangles.resize(dst);
	dst=0;
	loopi(0,_vertices.size())
	if(_vertices[i].tcount)
	{
		_vertices[i].tstart=dst;
		_vertices[dst].p=_vertices[i].p;
		dst++;
	}
	loopi(0,_triangles.size())
	{
		Triangle &t=_triangles[i];
		loopj(0,3)t.v[j]=_vertices[t.v[j]].tstart;
	}
	_vertices.resize(dst);
}

	// Error between vertex and Quadric

double FastQuadricDecimator::vertex_error(SymetricMatrix q, double x, double y, double z)
{
	return   q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x + q[4]*y*y
			+ 2*q[5]*y*z + 2*q[6]*y + q[7]*z*z + 2*q[8]*z + q[9];
}

// Error for one edge

double FastQuadricDecimator::calculate_error(int id_v1, int id_v2, vec3f &p_result)
{
	// compute interpolated vertex

	SymetricMatrix q = _vertices[id_v1].q + _vertices[id_v2].q;
	bool   border = _vertices[id_v1].border & _vertices[id_v2].border;
	double error=0;
	double det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
	if ( det != 0 && !border )
	{

		// q_delta is invertible
		p_result.x = -1/det*(q.det(1, 2, 3, 4, 5, 6, 5, 7 , 8));	// vx = A41/det(q_delta)
		p_result.y =  1/det*(q.det(0, 2, 3, 1, 5, 6, 2, 7 , 8));	// vy = A42/det(q_delta)
		p_result.z = -1/det*(q.det(0, 1, 3, 1, 4, 6, 2, 5,  8));	// vz = A43/det(q_delta)

		error = vertex_error(q, p_result.x, p_result.y, p_result.z);
	}
	else
	{
		// det = 0 -> try to find best result
		vec3f p1=_vertices[id_v1].p;
		vec3f p2=_vertices[id_v2].p;
		vec3f p3=(p1+p2)/2;
		double error1 = vertex_error(q, p1.x,p1.y,p1.z);
		double error2 = vertex_error(q, p2.x,p2.y,p2.z);
		double error3 = vertex_error(q, p3.x,p3.y,p3.z);
		error = min(error1, min(error2, error3));
		if (error1 == error) p_result=p1;
		if (error2 == error) p_result=p2;
		if (error3 == error) p_result=p3;
	}
	return error;
}

char *FastQuadricDecimator::trimwhitespace(char *str)
{
	char *end;

	// Trim leading space
	while(isspace((unsigned char)*str)) str++;

	if(*str == 0)  // All spaces?
	return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while(end > str && isspace((unsigned char)*end)) end--;

	// Write new null terminator
	*(end+1) = 0;

	return str;
}


