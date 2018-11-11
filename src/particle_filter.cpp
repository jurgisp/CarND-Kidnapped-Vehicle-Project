/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::ostream& operator << (std::ostream &o, const Particle &p) {
  o << "(" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
  return o;
}

std::ostream& operator << (std::ostream &o, const LandmarkObs &p) {
  o << "(" << p.id << ", " << p.x << ", " << p.y << ")" << endl;
  return o;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (auto i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

  is_initialized = true;
  cout << "Initialized " << particles.size() << " particles" << endl;
}

void ParticleFilter::prediction(double delta_t, double velocity, double yaw_rate,
                                double std_velocity, double std_yaw_rate) {

  double sum_x, sum_y;

  default_random_engine gen;
  normal_distribution<double> dist_v(0, std_velocity);
  normal_distribution<double> dist_yr(0, std_yaw_rate);

  for (auto i = 0; i < num_particles; i++) {

    double v = velocity + dist_v(gen);
    double yr = yaw_rate + dist_yr(gen);

    auto p = particles.at(i);
    if (yr == 0) {
      p.x += v * delta_t * cos(p.theta);
      p.y += v * delta_t * sin(p.theta);
    }
    else {
      p.x += v / yr * (sin(p.theta + yr * delta_t) - sin(p.theta));
      p.y += v / yr * (-cos(p.theta + yr * delta_t) + cos(p.theta));
      p.theta += yr * delta_t;
    }

    sum_x += p.x;
    sum_y += p.y;

    particles.at(i) = p;
  }

  //cout << "avg (x, y) = (" << sum_x/num_particles << ", " << sum_y/num_particles << ")" << endl;
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

  for (auto i = 0; i < num_particles; i++) {

    auto part = particles.at(i);

    double weight = 1.0;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    for (size_t j = 0; j < observations.size(); j++) {

      auto obs = observations.at(j);

      // Transform observation to map coordinates
      double x = part.x + obs.x * cos(part.theta) - obs.y * sin(part.theta);
      double y = part.y + obs.x * sin(part.theta) + obs.y * cos(part.theta);

      // Find closest landmark
      double min_dist = 1e10;
      int min_id = 0;
      double min_x = 0;
      double min_y = 0;

      for (size_t k = 0; k < map_landmarks.landmark_list.size(); k++) {
        auto lm = map_landmarks.landmark_list.at(k);
        double dist = sqrt((lm.x_f - x) * (lm.x_f - x) + (lm.y_f - y) * (lm.y_f - y));
        if (dist < min_dist) {
          min_dist = dist;
          min_id = lm.id_i;
          min_x = lm.x_f;
          min_y = lm.y_f;
        }
      }

      associations.push_back(min_id);
      sense_x.push_back(min_x);
      sense_y.push_back(min_y);

      double exponent= (x - min_x)*(x - min_x)/(2 * sig_x*sig_x) + (y - min_y)*(y - min_y)/(2 * sig_y*sig_y);
      double lm_weight = gauss_norm * exp(-exponent);
      weight = weight * lm_weight;
    }

    part.weight = weight;
    part.associations = associations;
    part.sense_x = sense_x;
    part.sense_y = sense_y;

    particles.at(i) = part;
  }
}

void ParticleFilter::resample() {

  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles.at(i).weight);
  }

  default_random_engine gen;
  discrete_distribution<> dist(weights.begin(), weights.end());
  vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles.at(dist(gen)));
  }

  particles = new_particles;
}


string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
