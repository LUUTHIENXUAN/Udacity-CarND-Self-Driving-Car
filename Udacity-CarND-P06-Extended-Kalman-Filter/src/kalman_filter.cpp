#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_          = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_          = F_ * P_ * Ft + Q_;
  
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
   // calculate y for the lidar sensor
  VectorXd y      = z - z_pred;
  
  /* There is no angle measurements sent by the Lidar
  */
  
  MatrixXd Ht     = H_.transpose();
  MatrixXd S      = H_ * P_ * Ht + R_;
  MatrixXd Si     = S.inverse();
  MatrixXd PHt    = P_ * Ht;
  MatrixXd K      = PHt * Si;
  
  //new estimate
  x_          = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I  = MatrixXd::Identity(x_size, x_size);
  P_          = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  
  float rho     = sqrt(px*px + py*py);
  float phi     = atan2(py , px); // already return value in -pi and pi
  
  if (rho < 0.0001) rho = 0.0001;
  float rho_dot = (px*vx + py*vy)/rho;
  
  // The radar sensor's output values in polar coordinates
  VectorXd Hx_(3);
  Hx_ << rho, phi, rho_dot;
  
  // calculate y for the radar sensor
  VectorXd y = z - Hx_;
  
  const float Pi2 = 2 * M_PI;
  
  // normalize phi in the y vector so that its angle is between -pi and pi
  // y[1] = atan2(sin(y[1]), cos(y[1]));
  
  if (y[1] > M_PI){
	  y[1] -= Pi2;
  }
   
  if (y[1] < -M_PI){
	  y[1] += Pi2; 
  }
  
  if ((y[1] < -M_PI) || (y[1] > M_PI)) {
	// print the output
	std::cout << "phi = " << y[1] << std::endl;  
  }
  
  MatrixXd Ht     = H_.transpose();
  MatrixXd S      = H_ * P_ * Ht + R_;
  MatrixXd Si     = S.inverse();
  MatrixXd PHt    = P_ * Ht;
  MatrixXd K      = PHt * Si;
  
  //new estimate
  x_          = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I  = MatrixXd::Identity(x_size, x_size);
  P_          = (I - K * H_) * P_;
  
}
