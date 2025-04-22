#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdlib>

// Параметры симуляции
const double BOX_SIZE       = 20e-9;
const int    NUM_PARTICLES  = 100000;
// Количество частиц Аргона
const int    N1             = NUM_PARTICLES / 2;
// ... Криптона
const int    N2             = NUM_PARTICLES - N1;
const double TIME_STEP      = 1e-15; // 1 фс
const int    STEPS          = 50000;
const int    GRID_SIZE      = 100;
const double BOLTZMANN      = 1.380649e-23;

// Масса Аргона
const double MASS1   = 39.948e-3 / 6.022e23;
// Эффективный радиус Аргона
const double RADIUS1 = 3.4e-10;
// То же самое для Криптона
const double MASS2   = 83.798e-3 / 6.022e23;
const double RADIUS2 = 3.7e-10;
const double INITIAL_TEMPERATURE = 300.0; // В Кельвинах

struct Vec2 {
    double x, y;
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s)     const { return {x * s, y * s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    double length2() const { return x*x + y*y; }
    double length()  const { return std::sqrt(length2()); }
    double dot(const Vec2& o) const { return x*o.x + y*o.y; }
};

struct Particle {
    Vec2 pos, vel;
    double mass, radius;
    int    species;  // 0 или 1
};

std::vector<Particle> particles;
std::vector<std::vector<int>> grid(GRID_SIZE * GRID_SIZE);
double wall_impulse = 0.0;

inline int gridIndex(int x, int y) {
    return y * GRID_SIZE + x;
}

void placeParticles() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist_pos(RADIUS2, BOX_SIZE - RADIUS2);

    particles.resize(NUM_PARTICLES);
    double v1 = std::sqrt(2.0 * BOLTZMANN * INITIAL_TEMPERATURE * N1);
    double v2 = std::sqrt(2.0 * BOLTZMANN * INITIAL_TEMPERATURE * N2);
    std::normal_distribution<double> dist_vel(0.0, 1.0);

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles[i].pos = { dist_pos(rng), dist_pos(rng) };
        if (i < N1) {
            // Первая компонента
            particles[i].mass    = MASS1;
            particles[i].radius  = RADIUS1;
            particles[i].species = 0;
            double sigma = std::sqrt(BOLTZMANN * INITIAL_TEMPERATURE / MASS1);
            particles[i].vel = { dist_vel(rng) * sigma, dist_vel(rng) * sigma };
            particles[i].vel = { dist_vel(rng) * sigma, dist_vel(rng) * sigma };
        } else {
            // Вторая компонента
            particles[i].mass    = MASS2;
            particles[i].radius  = RADIUS2;
            particles[i].species = 1;
            double sigma = std::sqrt(BOLTZMANN * INITIAL_TEMPERATURE / MASS2);
            particles[i].vel = { dist_vel(rng) * sigma, dist_vel(rng) * sigma };
            particles[i].vel = { dist_vel(rng) * sigma, dist_vel(rng) * sigma };
            
        }
        // particles[i].vel = { 0.0, 0.0 };
        // particles[i].vel = { dist_norm(rng) * sigma, dist_norm(rng) * sigma };
        // particles[0].vel = {v1, 0.0};
        // particles[N1].vel = {v2, 0.0};
    }
}

void updateGrid() {
    for (auto& cell : grid) cell.clear();
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int gx = std::clamp(int(particles[i].pos.x / BOX_SIZE * GRID_SIZE), 0, GRID_SIZE-1);
        int gy = std::clamp(int(particles[i].pos.y / BOX_SIZE * GRID_SIZE), 0, GRID_SIZE-1);
        grid[gridIndex(gx, gy)].push_back(i);
    }
}

void resolveCollision(Particle& a, Particle& b) {
    Vec2 delta = b.pos - a.pos;
    double dist = delta.length();
    double Rsum = a.radius + b.radius;
    if (dist >= Rsum || dist < 1e-12) return;

    Vec2 n = delta * (1.0 / dist);
    Vec2 v_rel = a.vel - b.vel;
    double v_rel_n = v_rel.dot(n);
    if (v_rel_n >= 0) return;  // расходятся

    // упругое столкновение
    double invM1 = 1.0 / a.mass;
    double invM2 = 1.0 / b.mass;
    double j = -(1.0 + 1.0) * v_rel_n / (invM1 + invM2);
    Vec2 J = n * j;

    a.vel += J * invM1;
    b.vel = b.vel - J * invM2;

    // убрать перекрытие
    double overlap = 0.5 * (Rsum - dist);
    a.pos = a.pos - n * overlap;
    b.pos += n * overlap;
}

void simulateStep() {
    wall_impulse = 0.0;
    // движение
    for (auto& p : particles) {
        p.pos += p.vel * TIME_STEP;
    }
    updateGrid();

    // столкновения между частицами
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        auto& a = particles[i];
        int gx = std::clamp(int(a.pos.x / BOX_SIZE * GRID_SIZE), 0, GRID_SIZE-1);
        int gy = std::clamp(int(a.pos.y / BOX_SIZE * GRID_SIZE), 0, GRID_SIZE-1);
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) {
            int nx = gx + dx, ny = gy + dy;
            if (nx < 0 || ny < 0 || nx >= GRID_SIZE || ny >= GRID_SIZE) continue;
            for (int j : grid[gridIndex(nx, ny)]) {
                if (j <= i) continue;
                resolveCollision(a, particles[j]);
            }
        }
    }

    // столкновения со стенками
    for (auto& p : particles) {
        if (p.pos.x < p.radius) {
            wall_impulse += std::abs(2.0 * p.mass * p.vel.x);
            p.vel.x *= -1; p.pos.x = p.radius;
        } else if (p.pos.x > BOX_SIZE - p.radius) {
            wall_impulse += std::abs(2.0 * p.mass * p.vel.x);
            p.vel.x *= -1; p.pos.x = BOX_SIZE - p.radius;
        }
        if (p.pos.y < p.radius) {
            wall_impulse += std::abs(2.0 * p.mass * p.vel.y);
            p.vel.y *= -1; p.pos.y = p.radius;
        } else if (p.pos.y > BOX_SIZE - p.radius) {
            wall_impulse += std::abs(2.0 * p.mass * p.vel.y);
            p.vel.y *= -1; p.pos.y = BOX_SIZE - p.radius;
        }
    }
}

void logState(std::ofstream& file, int step) {
    double E1 = 0, E2 = 0;
    double v_mean = 0.0;
    double vx2_1 = 0, vy2_1 = 0, cnt1 = 0;
    double vx2_2 = 0, vy2_2 = 0, cnt2 = 0;
    for (auto& p : particles) {
        double v2 = p.vel.length2();
        if (p.species == 0) {
            E1 += 0.5 * p.mass * v2;
            vx2_1 += p.vel.x*p.vel.x; vy2_1 += p.vel.y*p.vel.y; cnt1 += 1;
            // if (cnt1 < 500) {
            //     file2 << "," << particles[i].vel.x << "," << particles[i].vel.y
            // }
        } else {
            E2 += 0.5 * p.mass * v2;
            vx2_2 += p.vel.x*p.vel.x; vy2_2 += p.vel.y*p.vel.y; cnt2 += 1;
        }
    }
    vx2_1 /= cnt1; vy2_1 /= cnt1;
    vx2_2 /= cnt2; vy2_2 /= cnt2;

    double E_tot = E1 + E2;
    double T       = E_tot / NUM_PARTICLES / BOLTZMANN;
    double P_ideal = NUM_PARTICLES * BOLTZMANN * T / (BOX_SIZE*BOX_SIZE);
    double P_real  = wall_impulse / (BOX_SIZE*BOX_SIZE * TIME_STEP * 100);

    file << step*TIME_STEP << ","
         << E1 << "," << E2 << ","
         << vx2_1 << "," << vy2_1 << ","
         << vx2_2 << "," << vy2_2 << ","
         << T  << "," << P_ideal << "," << P_real << "," << wall_impulse;
    const int K = 100;  // Сколько частиц логировать
    for (int i = 0; i < K; ++i) {
        int j = rand() % (N1);
        file << "," << particles[j].vel.x << "," << particles[j].vel.y;
    }
    for (int i = 0; i < K; ++i) {
        int j = N1 + rand() % (N2);
        file << "," << particles[j].vel.x << "," << particles[j].vel.y;
    }
    file << '\n';
         
}

int main() {
    placeParticles();
    std::ofstream state_file("state4.csv");
    state_file << "time,E1,E2,vx2_1,vy2_1,vx2_2,vy2_2,temperature,pressure_ideal,pressure_real,wall_impulse";

    const int K = 100;
    for (int i = 0; i < K; ++i) {
        state_file << ",vx1_" << i << ",vy1_" << i;
    }
    for (int i = 0; i < K; ++i) {
        state_file << ",vx2_" << i << ",vy2_" << i;
    }
    state_file << '\n';
    
    for (int step = 0; step < STEPS; ++step) {
        simulateStep();
        if (step % 100 == 0) logState(state_file, step);
    }
    state_file.close();
    return 0;
}
