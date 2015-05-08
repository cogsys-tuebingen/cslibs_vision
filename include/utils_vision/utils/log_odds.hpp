#ifndef LOG_ODDS_HPP
#define LOG_ODDS_HPP

/// SYSTEM
#include <cmath>

namespace utils_vision {
template<typename Tp>
struct Odds {
    inline static Tp toProb(const Tp odds)
    {
        static Tp _1 = (Tp) 1;
        return _1 - _1 / (_1 - std::exp(odds));
    }
};

template<typename Tp>
struct Prob {
    inline static Tp toOdds(const Tp prob)
    {
        static Tp _1 = (Tp) 1;
        return std::log(prob / (_1 - prob));
    }
};

}

#endif // LOG_ODDS_HPP
