#include "jetson.hpp"
#include "nlohmann/json.hpp"
#include "pros/apix.h"
#include "pros/rtos.hpp"
#include <exception>
#include <iostream>
#include <optional>
#include <string>

using json = nlohmann::json;
using namespace std;

namespace serial {
    // Make most_recent_frame accessible globally
    std::optional<Frame> most_recent_frame = std::nullopt;

    std::optional<Frame> parseFrame(const json &j) {
        try {
            Frame f;
            f.flag = j.value("flag", "");

            if (j.contains("pose") && j["pose"].is_object()) {
                const auto &p = j["pose"];
                double x = p.value("x", 0.0);
                double y = p.value("y", 0.0);
                double theta = p.value("theta", 0.0);
                f.pose = std::make_tuple(x, y, theta);
            } else {
                return std::nullopt;
            }

            // The JSON uses the key "stuff" for the detections
            if (j.contains("stuff") && j["stuff"].is_array()) {
                for (const auto &item : j["stuff"]) {
                    if (item.is_object()) {
                        Detection d;
                        d.x = item.value("x", 0.0);
                        d.y = item.value("y", 0.0);
                        d.z = item.value("z", 0.0);
                        d.cls = item.value("class", std::string{});
                        f.detections.push_back(d);
                    }
                }
            }

            return f;
        } catch (...) {
            // Catch any parsing errors that might occur
            return std::nullopt;
        }
    }

    std::optional<Frame> fetch_frame() {
        try {
            std::string line;
            if (std::getline(std::cin, line)) {
                json j = json::parse(line, nullptr, false);
                if (j.is_discarded()) {
                    std::cerr << "Invalid JSON received" << std::endl;
                    return std::nullopt;
                }
                return parseFrame(j);
            }
        } catch (json::parse_error &e) {
            std::cerr << "JSON parsing error: " << e.what() << '\n'
                      << "exception id: " << e.id << '\n'
                      << "byte position of error: " << e.byte << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "An unexpected error occurred: " << e.what()
                      << std::endl;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
        }
        return std::nullopt;
    }


    void task_send() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL);

        while (true) {
            // TODO: get the actual data using chassis.getPose()
            printf("{\"x\": %f,\"y\": %f, \"theta\": %f}\r\n", 1.0, 2.0, 180.0);
            fflush(stdout);

            pros::delay(30);
        }
    }
    void task_receive() {
        pros::c::serctl(SERCTL_DISABLE_COBS, NULL); // for redundancy

        while (true) {
            most_recent_frame = fetch_frame();

            pros::delay(30);
        }
    }

} // namespace serial
