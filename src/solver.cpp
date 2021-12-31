#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, const std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#define ENABLE_DUMP
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() { return (time() - t - paused) * 1000.0; }
} timer;

/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; } // inclusive
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }



using pii = std::pair<int, int>;

constexpr int N = 30;
constexpr int NN = N * N;

struct HamiltonianPathOn2DGrid {

    static constexpr int di[] = { 0, -1, 0, 1 };
    static constexpr int dj[] = { 1, 0, -1, 0 };

    Xorshift rnd;
    std::vector<int> dirs;

    int N;
    std::vector<std::pair<int, int>> points;

    HamiltonianPathOn2DGrid(int N, unsigned seed = 0) : N(N) {
        rnd.set_seed(seed);
        dirs = { 0, 1, 2, 3 };
    }

    bool is_inside(int i, int j) const {
        return 0 <= i && i < N && 0 <= j && j < N;
    }

    int find(int i, int j) const {
        for (int k = 0; k < points.size(); k++) {
            if (i != points[k].first || j != points[k].second) continue;
            return k;
        }
        return -1;
    }

    void add_point(int i, int j) {
        points.emplace_back(i, j);
    }

    void initialize() {
        for (int i = 0; i < N; i++) {
            if (i % 2 == 0) {
                for (int j = 0; j < N; j++) {
                    add_point(i, j);
                }
            }
            else {
                for (int j = N - 1; j >= 0; j--) {
                    add_point(i, j);
                }
            }
        }
    }

    void move_reverse() {
        std::reverse(points.begin(), points.end());
    }

    int move_backbite() {
        shuffle_vector(dirs, rnd);
        auto [ei, ej] = points.back();
        for (int d : dirs) {
            int ni = ei + di[d], nj = ej + dj[d];
            if (!is_inside(ni, nj)) continue;
            int idx = find(ni, nj);
            std::reverse(points.begin(), points.end());
            std::reverse(points.begin(), points.end() - idx - 1);
            return idx;
        }
        return -1;
    }

    void undo_backbite(int idx) {
        std::reverse(points.begin(), points.end() - idx - 1);
        std::reverse(points.begin(), points.end());
    }

#ifdef HAVE_OPENCV_HIGHGUI
    void show(int delay = 0) const {
        int grid_size = 500 / N;
        int img_size = grid_size * N;

        auto to_img_coord = [&](int i, int j) {
            return cv::Point(grid_size * j + grid_size / 2, grid_size * i + grid_size / 2);
        };

        auto get_color = [](double ratio) {
            cv::Scalar color;
            if (ratio < 0.5) {
                // blue to purple
                int val = std::round(ratio * 255 / 0.5);
                color = cv::Scalar(255, 0, val);
            }
            else {
                int val = std::round((ratio - 0.5) * 255 / 0.5);
                color = cv::Scalar(255 - val, 0, 255);
            }
            return color;
        };

        //int line_width = std::max(1, grid_size / 4);
        int line_width = 2;
        cv::Mat_<cv::Vec3b> img(img_size, img_size, cv::Vec3b(255, 255, 255));
        for (int k = 1; k < points.size(); k++) {
            auto [i1, j1] = points[k - 1];
            auto [i2, j2] = points[k];
            cv::arrowedLine(img, to_img_coord(i1, j1), to_img_coord(i2, j2), get_color(double(k) / points.size()), line_width, 8, 0, 0.4);
        }

        cv::imshow("img", img);
        cv::waitKey(delay);
    }
#endif

};

struct TestCase;
using TestCasePtr = std::shared_ptr<TestCase>;
struct TestCase {
    int A[N][N];
    static TestCasePtr create(unsigned seed) {
        Xorshift rnd;
        rnd.set_seed(seed);
        auto tc = std::make_shared<TestCase>();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                tc->A[i][j] = rnd.next_int(1, 100);
            }
        }
        return tc;
    }
    static TestCasePtr load(std::istream& in) {
        auto tc = std::make_shared<TestCase>();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                in >> tc->A[i][j];
            }
        }
        return tc;
    }
};

struct State {

    using Path = std::vector<pii>;

    int A[N][N];

    int remains;
    Path path;
    std::vector<std::vector<pii>> moves;

    State(TestCasePtr tc) {
        std::memcpy(A, tc->A, sizeof(int) * NN);
        remains = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                remains += A[i][j];
            }
        }
    }

    void set_path(const Path& path) {
        this->path = path;
    }

    bool is_keypoint(int idx) const {
        if (A[path[idx].first][path[idx].second] == 0) return false;
        if (idx == 0) return true;
        auto [i, j] = path[idx];
        auto [pi, pj] = path[idx - 1];
        return A[pi][pj] <= A[i][j];
    }

    void move_along_path(int idx) {
        std::vector<pii> ms;
        auto [pi, pj] = path[idx];
        ms.emplace_back(pi, pj);
        A[pi][pj]--;
        remains--;
        for (int k = idx + 1; k < path.size(); k++) {
            auto [i, j] = path[k];
            if (!A[i][j] || A[i][j] != A[pi][pj]) break;
            ms.emplace_back(i, j);
            A[i][j]--;
            remains--;
            pi = i; pj = j;
        }
        moves.push_back(ms);
    }

    int solve() {
        for (int idx = path.size() - 1; idx >= 0; idx--) {
            while (is_keypoint(idx)) {
                move_along_path(idx);
            }
        }
        return 100000 - (int)moves.size();
    }

    void output(std::ostream& out) const {
        std::ostringstream oss;
        for (const auto& ms : moves) {
            for (const auto& [i, j] : ms) {
                oss << i + 1 << ' ' << j + 1 << '\n';
            }
        }
        out << oss.str();
    }

};

int evaluate(TestCasePtr tc, const std::vector<pii>& path) {
    const auto& A = tc->A;
    int cost = A[path[0].first][path[0].second];
    for (int idx = 1; idx < path.size(); idx++) {
        auto [i, j] = path[idx];
        auto [pi, pj] = path[idx - 1];
        cost += std::max(0, A[i][j] - A[pi][pj] + 1);
    }
    return cost;
}

int solve(TestCasePtr tc, std::ostream& out, bool no_output = false) {

    auto get_temp = [](double start_temp, double end_temp, double now_time, double end_time) {
        return end_temp + (start_temp - end_temp) * (end_time - now_time) / end_time;
    };

    HamiltonianPathOn2DGrid grid(N);
    grid.initialize();

    int prev_cost = evaluate(tc, grid.points);
    int min_cost = prev_cost;
    auto best_path = grid.points;

    int loop = 0;
    double start_time = timer.elapsed_ms(), now_time, end_time = 9000;
    while ((now_time = timer.elapsed_ms()) < end_time) {
        loop++;
        int idx = grid.move_backbite();
        if (idx == -1) continue;
        int cost = evaluate(tc, grid.points);
        int diff = cost - prev_cost;
        double temp = get_temp(50.0, 0.0, now_time - start_time, end_time - start_time);
        double prob = exp(-diff / temp);
        if (prob < rnd.next_double()) {
            grid.undo_backbite(idx);
        }
        else {
            prev_cost = cost;
            if (cost < min_cost) {
                min_cost = cost;
                best_path = grid.points;
            }
        }
        if (!(loop & 65535)) {
            dump(loop, min_cost, cost);
        }
    }
    dump(loop, min_cost);    

    State state(tc);
    state.set_path(best_path);
    state.solve();

    if (!no_output) state.output(out);

    return 100000 - (int)state.moves.size();
}

int main() {

#ifdef _MSC_VER
    unsigned seed = 0;
    auto tc = TestCase::create(seed);
    int score = solve(tc, std::cout, true);
    dump(timer.elapsed_ms(), score);
#else
    auto tc = TestCase::load(std::cin);
    int score = solve(tc, std::cout);
    dump(timer.elapsed_ms(), score);
#endif

    return 0;
}