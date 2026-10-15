#include "kernel.hpp"
#include <iostream>
#include <iomanip>

int main() {
    pressure_lif::Kernel baseline(false), proposed(true);
    int x, y, z, up, down, contact;
    std::cout << std::setprecision(17);
    while (std::cin >> x >> y >> z >> up >> down >> contact) {
        const std::array<bool, 3> axes{{x != 0, y != 0, z != 0}};
        const auto b = baseline.step(axes, up != 0, down != 0, contact != 0);
        const auto p = proposed.step(axes, up != 0, down != 0, contact != 0);
        std::cout << b.spike << ' ' << p.spike << ' ' << p.acceleration_fired << ' '
                  << p.inhibition << ' ' << p.excitation << ' '
                  << p.inhibition_500ms << ' ' << p.excitation_500ms << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
