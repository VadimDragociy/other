#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

const double BOX_SIZE = 40.0;
const int NUM_PARTICLES = 10000;
const double TIME_STEP = 0.001;
const int STEPS = 5000;
const double RADIUS = 0.3;
const double MASS = 1.0;
const double INITIAL_TEMPERATURE = 100.0;
const int GRID_SIZE = 100;
const double BOLTZMANN = 1.0;

struct Vec2 {
  double x, y;
  Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
  Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
  Vec2 operator*(double s) const { return {x * s, y * s}; }
  Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
  Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
  double length2() const { return x * x + y * y; }
  double length() const { return std::sqrt(length2()); }
  double dot(const Vec2& o) const { return x * o.x + y * o.y; }
};

struct Particle {
  Vec2 pos;
  Vec2 vel;
};

std::vector<Particle> particles;
std::vector<std::vector<int>> grid(GRID_SIZE * GRID_SIZE);
double wall_impulse = 0.0;

int gridIndex(int x, int y) {
  return y * GRID_SIZE + x;
}

void placeParticles() {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist_pos(RADIUS, BOX_SIZE - RADIUS);

  particles.resize(NUM_PARTICLES);
  for (int i = 0; i < NUM_PARTICLES; ++i) {
    particles[i].pos = {dist_pos(rng), dist_pos(rng)};
    particles[i].vel = {0.0, 0.0};
  }

  double v0 = std::sqrt(2.0 * BOLTZMANN * INITIAL_TEMPERATURE * NUM_PARTICLES);
  particles[0].vel = {v0, 0.0};
}

void updateGrid() {
  for (auto& cell : grid) cell.clear();
  for (int i = 0; i < NUM_PARTICLES; ++i) {
    int gx = static_cast<int>(particles[i].pos.x / BOX_SIZE * GRID_SIZE);
    int gy = static_cast<int>(particles[i].pos.y / BOX_SIZE * GRID_SIZE);
    gx = std::clamp(gx, 0, GRID_SIZE - 1);
    gy = std::clamp(gy, 0, GRID_SIZE - 1);
    grid[gridIndex(gx, gy)].push_back(i);
  }
}

void resolveCollision(Particle& a, Particle& b) {
  Vec2 delta = b.pos - a.pos;
  double dist2 = delta.length2();
  if (dist2 < 1e-12) return;
  double dist = std::sqrt(dist2);
  if (dist >= 2 * RADIUS) return;

  Vec2 n = delta * (1.0 / dist);
  Vec2 v_rel = a.vel - b.vel;
  double v_rel_n = v_rel.dot(n);
  if (v_rel_n > -1e-8) return;

  Vec2 dv = n * v_rel_n;
  a.vel -= dv;
  b.vel += dv;

  double overlap = 0.5 * (2 * RADIUS - dist);
  a.pos -= n * overlap;
  b.pos += n * overlap;
}

void simulateStep() {
  wall_impulse = 0.0;

  for (Particle& p : particles) {
    p.pos += p.vel * TIME_STEP;
  }

  updateGrid();

  for (int i = 0; i < NUM_PARTICLES; ++i) {
    Particle& a = particles[i];
    int gx = static_cast<int>(a.pos.x / BOX_SIZE * GRID_SIZE);
    int gy = static_cast<int>(a.pos.y / BOX_SIZE * GRID_SIZE);

    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        int nx = gx + dx, ny = gy + dy;
        if (nx < 0 || ny < 0 || nx >= GRID_SIZE || ny >= GRID_SIZE) continue;
        for (int j : grid[gridIndex(nx, ny)]) {
          if (j <= i) continue;
          resolveCollision(a, particles[j]);
        }
      }
    }
  }

  for (Particle& p : particles) {
    if (p.pos.x < RADIUS) {
      wall_impulse += std::abs(2 * MASS * p.vel.x);
      p.vel.x *= -1;
      p.pos.x = RADIUS;
    } else if (p.pos.x > BOX_SIZE - RADIUS) {
      wall_impulse += std::abs(2 * MASS * p.vel.x);
      p.vel.x *= -1;
      p.pos.x = BOX_SIZE - RADIUS;
    }

    if (p.pos.y < RADIUS) {
      wall_impulse += std::abs(2 * MASS * p.vel.y);
      p.vel.y *= -1;
      p.pos.y = RADIUS;
    } else if (p.pos.y > BOX_SIZE - RADIUS) {
      wall_impulse += std::abs(2 * MASS * p.vel.y);
      p.vel.y *= -1;
      p.pos.y = BOX_SIZE - RADIUS;
    }
  }
}

void logState(std::ofstream& file, int step) {
  double E_kin = 0.0;
  double v_mean = 0.0;
  double v2_mean = 0.0;
  double vx2 = 0.0, vy2 = 0.0;

  for (const auto& p : particles) {
    double v2 = p.vel.length2();
    E_kin += 0.5 * MASS * v2;
    v_mean += std::sqrt(v2);
    v2_mean += v2;
    vx2 += p.vel.x * p.vel.x;
    vy2 += p.vel.y * p.vel.y;
  }

  v_mean /= NUM_PARTICLES;
  v2_mean /= NUM_PARTICLES;
  vx2 /= NUM_PARTICLES;
  vy2 /= NUM_PARTICLES;
  double T = (E_kin / NUM_PARTICLES) / BOLTZMANN;
  double P_ideal = (NUM_PARTICLES * std::sqrt(v2_mean) * MASS) / (2.0 * BOX_SIZE * BOX_SIZE);
  double P_real = wall_impulse / (BOX_SIZE * BOX_SIZE * TIME_STEP * 100);

  double vx0 = particles[0].vel.x;
  double vy0 = particles[0].vel.y;

  file << step * TIME_STEP << ","
       << E_kin << "," << v_mean << "," << std::sqrt(v2_mean) << ","
       << vx2 << "," << vy2 << "," << T << "," << P_ideal << "," << P_real << ","
       << vx0 << "," << vy0;

  std::cout << step * TIME_STEP << ","
       << E_kin << "," << v_mean << "," << std::sqrt(v2_mean) << ","
       << vx2 << "," << vy2 << "," << T << "," << P_ideal << "," << P_real << ","
       << vx0 << "," << vy0 << "\n";

  for (int i = 0; i < 100; ++i) {
    file << "," << particles[i].vel.x << "," << particles[i].vel.y;
  }
  file << "\n";
}

int main() {
  placeParticles();
  std::ofstream state_file("state1.csv");
  state_file << "time,kinetic_energy,mean_speed,rms_speed,vx2_mean,vy2_mean,temperature,pressure_ideal,pressure_real,vx0,vy0";

  for (int i = 0; i < 100; ++i) {
    state_file << ",vx_" << i << ",vy_" << i;
  }
  state_file << "\n";

  for (int step = 0; step < STEPS; ++step) {
    simulateStep();
    if (step % 100 == 0) {
      logState(state_file, step);
    }
  }

  state_file.close();
  return 0;
}
