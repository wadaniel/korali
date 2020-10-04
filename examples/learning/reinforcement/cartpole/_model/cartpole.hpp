//
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//  Modified for use in Korali

#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <functional>
#define SWINGUP 0

std::mt19937 _randomGenerator;

// Julien Berland, Christophe Bogey, Christophe Bailly,
// Low-dissipation and low-dispersion fourth-order Runge-Kutta algorithm,
// Computers & Fluids, Volume 35, Issue 10, December 2006, Pages 1459-1463, ISSN 0045-7930,
// http://dx.doi.org/10.1016/j.compfluid.2005.04.003
template <typename Func, typename Vec>
Vec rk46_nl(float t0, float dt, Vec u0, Func&& Diff)
{
  static constexpr float a[] = {0.000000000000, -0.737101392796,
            -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
  static constexpr float b[] = {0.032918605146,  0.823256998200,
             0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
  static constexpr float c[] = {0.000000000000,  0.032918605146,
             0.249351723343,  0.466911705055,  0.582030414044,  0.847252983783};
  static constexpr int s = 6;
  Vec w;
  Vec u(u0);
  float t;

  for (int i=0; i<s; ++i)
  {
    t = t0 + dt*c[i];
    w = w*a[i] + Diff(u, t)*dt;
    u = u + w*b[i];
  }
  return u;
}

struct Vec4
{
  float y1, y2, y3, y4;

  Vec4(float _y1=0, float _y2=0, float _y3=0, float _y4=0) :
    y1(_y1), y2(_y2), y3(_y3), y4(_y4) {};

  Vec4 operator*(float v) const
  {
    return Vec4(y1*v, y2*v, y3*v, y4*v);
  }

  Vec4 operator+(const Vec4& v) const
  {
    return Vec4(y1+v.y1, y2+v.y2, y3+v.y3, y4+v.y4);
  }
};

struct CartPole
{
  const float mp = 0.1;
  const float mc = 1;
  const float l = 0.5;
  const float g = 9.81;
  const float dt = 4e-4;
  const int nsteps = 50;
  // const float dt = 1e-6;   // emulate expensive application
  // const int nsteps = 20000; //
  int step=0;
  Vec4 u;
  float F=0, t=0;

 void reset()
 {
  #if SWINGUP
     std::uniform_real_distribution<float> dist(-.1,.1);
  #else
     std::uniform_real_distribution<float> dist(-0.05,0.05);
  #endif
  u = Vec4(dist(_randomGenerator), dist(_randomGenerator), dist(_randomGenerator), dist(_randomGenerator));
    step = 0;
  F = 0;
    t = 0;
 }

  bool is_failed()
  {
    #if SWINGUP
      return std::fabs(u.y1)>2.4;
    #else
      return std::fabs(u.y1)>2.4 || std::fabs(u.y3)>M_PI/15;
    #endif
  }
  bool is_over()
  {
    #if SWINGUP
      return std::fabs(u.y1)>2.4;
    #else
      return std::fabs(u.y1)>2.4 || std::fabs(u.y3)>M_PI/15;
    #endif
  }

  int advance(std::vector<float> action)
  {
    F = action[0];
    step++;
    for (int i=0; i<nsteps; i++) {
      u = rk46_nl(t, dt, u, std::bind(&CartPole::Diff,
                                      this,
                                      std::placeholders::_1,
                                      std::placeholders::_2) );
      t += dt;
      if( is_over() ) return 1;
    }
    return 0;
  }

 std::vector<float> getState()
 {
  std::vector<float> state(5);
  state[0] = u.y1;
  state[1] = u.y2;
  state[2] = u.y4;
  state[3] = std::cos(u.y3);
  state[4] = std::sin(u.y3);
  return state;
 }

  float getReward()
  {
    #if SWINGUP
      float angle = std::fmod(u.y3, 2*M_PI);
      angle = angle<0 ? angle+2*M_PI : angle;
      return std::fabs(angle-M_PI)<M_PI/6 ? 1 : 0;
    #else
      //return -1*( fabs(u.y3)>M_PI/15 || fabs(u.y1)>2.4 );
      return 1 - ( std::fabs(u.y3)>M_PI/15 || std::fabs(u.y1)>2.4 );
    #endif
  }

  Vec4 Diff(Vec4 _u, float _t)
  {
    Vec4 res;

    const float cosy = std::cos(_u.y3), siny = std::sin(_u.y3);
    const float w = _u.y4;
    #if SWINGUP
      const float fac1 = 1/(mc + mp * siny*siny);
      const float fac2 = fac1/l;
      res.y2 = fac1*(F + mp*siny*(l*w*w + g*cosy));
      res.y4 = fac2*(-F*cosy -mp*l*w*w*cosy*siny -(mc+mp)*g*siny);
    #else
      const float totMass = mp+mc;
      const float fac2 = l*(4.0/3 - (mp*cosy*cosy)/totMass);
      const float F1 = F + mp * l * w * w * siny;
      res.y4 = (g*siny - F1*cosy/totMass)/fac2;
      res.y2 = (F1 - mp*l*res.y4*cosy)/totMass;
    #endif
    res.y1 = _u.y2;
    res.y3 = _u.y4;
    return res;
  }
};
