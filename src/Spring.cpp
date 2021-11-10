#include "Spring.h"
#include "Particle.h"

using namespace std;
using namespace Eigen;

Spring::Spring(shared_ptr<Particle> p0, shared_ptr<Particle> p1) :
	E(1.0)
{
	assert(p0);
	assert(p1);
	assert(p0 != p1);
	this->p0 = p0;
	this->p1 = p1;
	Vector3d x0 = p0->x;
	Vector3d x1 = p1->x;
	Vector3d dx = x1 - x0;
	L = dx.norm();
	assert(L > 0.0);
}

Spring::~Spring()
{
	
}

Eigen::VectorXd Spring::computeForce()
{
	VectorXd deltaX(3);
	deltaX = p1->x - p0->x;
	double l = deltaX.norm();
	return ((E * (l - L) / l) * deltaX);
}

Eigen::MatrixXd Spring::computeStiffness()
{
	MatrixXd ks(3, 3);
	VectorXd deltaX(3);
	deltaX = p1->x - p0->x;
	double l = deltaX.norm();
	MatrixXd weird_delta_x = deltaX * deltaX.transpose();
	double dx_double = deltaX.transpose() * deltaX;
	MatrixXd I(3,3);
	I.setIdentity();
	ks = (E / (l * l)) * ((1 - (l - L) / l) * weird_delta_x + ((l - L) / l * dx_double) * I);
	return ks;
}
