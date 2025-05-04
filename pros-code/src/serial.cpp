#include "pros/apix.h" // Include for serctl
#include "receiver.hpp"
#include <cstdio>

namespace serial {
    void start() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL);
        // TODO: get the actual starting position from the auton selector or smth
        double ix = 1.0;
        double iy = 2.0;
        double theta = 180.0;

        printf("{\"x\": %f,\"y\": %f, \"theta\": %f}\r\n", ix, iy, theta);
        fflush(stdout);
    }

    void task() {
        while (true) {
            // TODO: get the actual imu reading from the imu
            printf("{\"x\": %f,\"y\": %f, \"theta\": %f}\r\n", 1.0, 2.0, 180.0);
            fflush(stdout);

            most_recent_frame = fetch_frame();

            pros::delay(20);
        }
    }
} // namespace serial
