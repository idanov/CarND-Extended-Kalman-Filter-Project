#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

auto crop = [](float x, float lower){ return fabs(x) > fabs(lower) ? x : lower; };
float lim = 1e-2;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  // laser measurement matrix
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // initial state covariance matrix P
  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  // the initial transition matrix F
  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 1, 0,
       0, 1, 0, 1,
       0, 0, 1, 0,
       0, 0, 0, 1;

  // initial process covariance matrix Q
  MatrixXd Q = MatrixXd(4, 4);
  Q << 0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0;

  // initial state
  VectorXd x = VectorXd(4);
  x << 0, 0, 0, 0;

  // initialize kalman filter
  ekf_.Init(x, P, F, H_laser_, R_laser_, Q);

  // set noise
  noise_ax = 9;
  noise_ay = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 0, 0, 0, 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // get measurements from Radar
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      // calculate cos/sin of phi
      float cos_phi = cos(phi);
      float sin_phi = sin(phi);

      // convert from polar to cartesian coordinates
      float px = crop(rho * cos_phi, lim);
      float py = crop(-rho * sin_phi, lim);
      float vx = rho_dot * cos_phi;
      float vy = -rho_dot * sin_phi;

      // initialize state
      ekf_.x_ << px, py, vx, vy;
      // set initial timestamp
      previous_timestamp_ = measurement_pack.timestamp_;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // initialize position state wiht Lidar measurements
      float px = crop(measurement_pack.raw_measurements_(0), lim);
      float py = crop(measurement_pack.raw_measurements_(1), lim);

      ekf_.x_ << px, py, 0, 0;
      // set initial timestamp
      previous_timestamp_ = measurement_pack.timestamp_;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	// dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // make sure dt is not 0
  dt = crop(dt, lim * lim);

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // update the state transition matrix F according to the new elapsed time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // update the process noise covariance matrix.
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax,               0, dt_3/2*noise_ax,               0,
                           0, dt_4/4*noise_ay,               0, dt_3/2*noise_ay,
             dt_3/2*noise_ax,               0,   dt_2*noise_ax,               0,
                           0, dt_3/2*noise_ay,               0,   dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
