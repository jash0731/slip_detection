#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pressure_lif {
struct Neuron {
    double beta, threshold, voltage = 0.;
    int refractory, remaining = 0;
    Neuron(double tau, int ticks, double limit = 1.)
        : beta(std::exp(-5. / tau)), threshold(limit), refractory(ticks) {}
    void reset() { voltage = 0.; remaining = 0; }
    bool step(double drive) {
        if (remaining) { --remaining; voltage = 0.; return false; }
        voltage = beta * voltage + drive;
        if (voltage >= threshold) { reset(); remaining = refractory - 1; return true; }
        return false;
    }
};
struct Result {
    bool spike = false, acceleration_fired = false;
    double inhibition = 0., excitation = 0., inhibition_500ms = 0., excitation_500ms = 0.;
};
class Kernel {
    std::array<Neuron, 3> axes{{Neuron(25., 2), Neuron(25., 2), Neuron(25., 2)}};
    std::array<Neuron, 2> pressure{{Neuron(30., 5), Neuron(30., 5)}};
    Neuron output{50., 10};
    std::array<double, 100> inhibition{}, excitation{};
    long tick = 0, last_acceleration = -1;
    bool enabled;
public:
    explicit Kernel(bool pressure_enabled) : enabled(pressure_enabled) {}
    void reset() {
        for (auto &n : axes) n.reset();
        for (auto &n : pressure) n.reset();
        output.reset(); inhibition.fill(0.); excitation.fill(0.);
        tick = 0; last_acceleration = -1;
    }
    Result step(const std::array<bool, 3> &acceleration, bool increase, bool decrease, bool contact) {
        if (!contact) { reset(); return {}; }
        Result result;
        const auto slot = static_cast<std::size_t>(tick % 100);
        inhibition[slot] = excitation[slot] = 0.;
        int fired = 0;
        for (std::size_t i = 0; i < axes.size(); ++i)
            fired += axes[i].step(acceleration[i] ? 1.1 : 0.) ? 1 : 0;
        const bool up = enabled && pressure[0].step(increase ? 1.1 : 0.);
        const bool down = enabled && pressure[1].step(decrease ? 1.1 : 0.);
        const bool recent = last_acceleration >= 0 && tick > last_acceleration && tick - last_acceleration <= 20;
        for (int offset = 1; offset < 100; ++offset) {
            const auto i = (slot + offset) % 100;
            result.inhibition_500ms += inhibition[i];
            result.excitation_500ms += excitation[i];
        }
        result.inhibition = enabled && up ? std::min(.1, std::max(0., .25 - result.inhibition_500ms)) : 0.;
        result.excitation = enabled && down && recent ? std::min(.1, std::max(0., .25 - result.excitation_500ms)) : 0.;
        inhibition[slot] = result.inhibition; excitation[slot] = result.excitation;
        result.inhibition_500ms += result.inhibition;
        result.excitation_500ms += result.excitation;
        result.spike = output.step(.4 * fired + result.excitation - result.inhibition);
        result.acceleration_fired = fired > 0;
        if (fired) last_acceleration = tick;
        ++tick;
        return result;
    }
};
}
