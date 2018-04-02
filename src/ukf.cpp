#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  
  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  
  P_ << 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0,.5, 0, 0,
        0, 0, 0,.5, 0,
        0, 0, 0, 0,.5;
  

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.8 ;// 30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7 ;// 30
  
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // State dimension
  n_x_     = 5;

  // Augmented state dimension
  n_aug_   = 7;
  
  //set measurement dimension, radar can measure r, phi, and r_dot
  n_z  = 3;

  // Sigma point spreading parameter
  lambda_  = 3 - n_aug_;
  
  //set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  // set weights
  weights_(0)       = lambda_/(lambda_+n_aug_);
   
  for (int i=1; i<2*n_aug_+1; i++) {
	  weights_(i) = 0.5/(n_aug_+lambda_);
  }
  
  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // previous time
  previous_timestamp_ = 0;
  
  // initializing R matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  
  R_laser_ << std_laspx_*std_laspx_,                     0,
                                  0, std_laspy_*std_laspy_;
                                  
  R_radar_ << std_radr_*std_radr_,                       0,                    0,
                                0, std_radphi_*std_radphi_,                    0,
                                0,                       0,std_radrd_*std_radrd_;
  
  // initializing h matrice
  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;
}

UKF::~UKF() {}

const float Pi2 = 2 * M_PI;
void angle_normalization(double angle){
    if (angle > M_PI){
        angle -= Pi2;
    }
    if (angle < -M_PI){
        angle += Pi2;
    }
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
   /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
	  
    x_ << 0.001, 0.001, 0.001, 0.001, 0.001;
       
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */      
      float rho     = meas_package.raw_measurements_[0];
      float phi     = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];
      
      x_ << rho * cos(phi), rho * sin(phi), 0.0, 0.0, 0.0;
      
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      
      /**
      Initialize state.
      */
      
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
    }
    
    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
    
  }
  
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
   //compute the time elapsed between the current and previous measurements
   
   float delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
   Prediction(delta_t);
   
   /*****************************************************************************
   *  Update
   ****************************************************************************/
   
   if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    
    // Radar updates
    R_ = R_radar_;
    UpdateRadar(meas_package);
    
   } else {
	
    // Laser updates
    R_ = R_laser_;
    H_ = H_laser_;      
    UpdateLidar(meas_package);

   }
   previous_timestamp_ = meas_package.timestamp_;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  /*****************************************************************************
   *  Augmented Sigma Points
   ****************************************************************************/
  
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  //create augmented mean state
  x_aug << x_, 0.0, 0.0;
  
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  //create augmented sigma points
  //calculate square root of augmented P
  MatrixXd A = P_aug.llt().matrixL();
  
  //set first column of sigma point matrix
  Xsig_aug.col(0)  = x_aug;
  //set remaining sigma points
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
    
  }
  
  
  /*****************************************************************************
   *  Sigma Point Prediction
   ****************************************************************************/
  
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
        
    }

    double v_p    = v;
    double yaw_p  = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p  = v_p + nu_a*delta_t;

    yaw_p  = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
    
  }
  
  /*****************************************************************************
   *  Predict Mean And Covariance
   ****************************************************************************/
   
   //predicted state mean
   x_.fill(0.0);
   for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
	   x_ = x_ + weights_(i) * Xsig_pred_.col(i);
   }
   
   //predicted state covariance matrix
   P_.fill(0.0);
   for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
	   // state difference
	   VectorXd x_diff = Xsig_pred_.col(i) - x_;
	   
	   //angle normalization
	   angle_normalization(x_diff(3));
	   
	   P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
	   
   }
   
  
  /*****************************************************************************
   *  Debug
   ****************************************************************************/
  /*
  std::cout << "************************************************:" << std::endl;
  std::cout << "UKF::Prediction Debug:" << std::endl;
  
  std::cout << " x_aug = " << std::endl << x_aug << std::endl;
  std::cout << " P_aug = " << std::endl << P_aug << std::endl;
  std::cout << " Xsig_aug = " << std::endl << Xsig_aug << std::endl;
  std::cout << " A = " << std::endl << A << std::endl;
  
  //print result Augmented Sigma Points
  std::cout << " Xsig_aug = " << std::endl << Xsig_aug << std::endl;
  
  //print result Sigma Point Prediction
  std::cout << " Xsig_pred = " << std::endl << Xsig_pred_ << std::endl;
  
  //print result Predict Mean And Covariance
  std::cout << " Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << " Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //create vector for incoming radar measurement
  VectorXd z   = VectorXd(2);
  
  float px_  = meas_package.raw_measurements_[0];
  float py_  = meas_package.raw_measurements_[1];
  
  z <<px_,   
      py_;
  
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
  
  //print result
  //std::cout << "(Lidar)Updated state x: " << std::endl << x_ << std::endl;
  //std::cout << "(Lidar)Updated state covariance P: " << std::endl << P_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  
  /*****************************************************************************
   *  Predict Radar Measurement
  ****************************************************************************/
   
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);          //r
    if (Zsig(0,i) < 0.0001) Zsig(0,i) = 0.0001;   //divide by 0
    
    Zsig(1,i) = atan2(p_y,p_x);                   //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i);   //r_dot
    
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    angle_normalization(z_diff(1));

    S  += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  
  S  = S + R_;
  
  /*****************************************************************************
   *  Update State
  ****************************************************************************/
  
  //create vector for incoming radar measurement
  VectorXd z   = VectorXd(3);
  
  float ro     = meas_package.raw_measurements_[0];//rho in m
  float theta  = meas_package.raw_measurements_[1];//phi in rad
  float ro_dot = meas_package.raw_measurements_[2];//rho_dot in m/s
  
  z <<ro,   
      theta,   
      ro_dot;   

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    angle_normalization (z_diff(1));
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    angle_normalization (x_diff(3));
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  //angle normalization
  angle_normalization (z_diff(1));
   
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  
  /*****************************************************************************
   *  Debug
   ****************************************************************************/
  
  /*
  std::cout << "************************************************:" << std::endl;
  std::cout << "UKF::UpdateRadar Debug:" << std::endl;
   
  //print result Predict Radar Measurement 
  std::cout << " Zsig: " << std::endl << Zsig << std::endl;
  std::cout << " z_pred: " << std::endl << z_pred << std::endl;
  std::cout << " S: " << std::endl << S << std::endl;
  
  //print result Update State
  std::cout << " (Radar)Updated state x: " << std::endl << x_ << std::endl;
  std::cout << " (Radar)Updated state covariance P: " << std::endl << P_ << std::endl;
  */
}
