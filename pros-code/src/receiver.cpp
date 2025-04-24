#include <iostream>
#include <optional>
#include <string>
#include "receiver.hpp"
#include <exception>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;
namespace serial {
optional<Frame> parseFrame(const json& j) {
    Frame f;
    f.flag = j.value("flag", "");

    if (j.contains("pose") && j["pose"].is_object()) {
        const auto& p = j["pose"];
        double x     = p.value("x",     0.0);
        double y     = p.value("y",     0.0);
        double theta = p.value("theta", 0.0);
        f.poses.emplace_back(x, y, theta);
    } else {
        return nullopt;
    }

    if (j.contains("stuff") && j["stuff"].is_array()) {
        for (const auto& item : j["stuff"]) {
            if (item.is_object()) {
                Detection d;
                d.x     = item.value("x",           0.0);
                d.y     = item.value("y",           0.0);
                d.w     = item.value("width",       0.0);
                d.h     = item.value("height",      0.0);
                d.cls   = item.value("class",       std::string{});
                d.depth = item.value("depth",       0.0);
                d.conf  = item.value("confidence",  0.0);
                f.detections.push_back(d);
            } else {
                return nullopt;
            }
        }
    } else {
        return nullopt;
    }
    return f;
}

optional<Frame> fetch_frame() {
    try {
        json j;
        std::cin >> j;
        return parseFrame(j);
    } catch (json::parse_error& e) {
        std::cerr << "JSON parsing error: " << e.what() << '\n'
                  << "exception id: " << e.id << '\n'
                  << "byte position of error: " << e.byte << std::endl;
        return nullopt;

    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return nullopt;

    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return nullopt;
    }
}

}
