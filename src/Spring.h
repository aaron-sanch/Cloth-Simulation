#pragma once
#ifndef Spring_H
#define Spring_H

#include <memory>
#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>

class Particle;

class Spring
{
public:
	Spring(std::shared_ptr<Particle> p0, std::shared_ptr<Particle> p1);
	virtual ~Spring();
	Eigen::VectorXd computeForce();
	Eigen::MatrixXd computeStiffness();
	
	std::shared_ptr<Particle> p0;
	std::shared_ptr<Particle> p1;
	double E;
	double L;
};

#endif
