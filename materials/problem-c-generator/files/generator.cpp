#include "testlib.h"

#include <iostream>
#include <vector>

struct DSU {
    std::vector<int> p;
    std::vector<int> r;
    explicit DSU(int n) : p(n), r(n, 0) {
        for (int i = 0; i < n; ++i) {
            p[i] = i;
        }
    }
    int find(int x) {
        if (p[x] == x) {
            return x;
        }
        p[x] = find(p[x]);
        return p[x];
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return false;
        }
        if (r[a] < r[b]) {
            std::swap(a, b);
        }
        p[b] = a;
        if (r[a] == r[b]) {
            ++r[a];
        }
        return true;
    }
};

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);

    const int N = 125;
    const int T = 10000;

    std::vector<std::string> grid(N, std::string(N, '#'));
    for (int i = 1; i < N; i += 2) {
        for (int j = 1; j < N; j += 2) {
            grid[i][j] = '.';
        }
    }

    const int r = N / 2;
    const int nodes = r * r;
    struct Edge {
        int a;
        int b;
    };
    std::vector<Edge> edges;
    edges.reserve((r - 1) * r * 2);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            int id = i * r + j;
            if (i + 1 < r) {
                edges.push_back({id, (i + 1) * r + j});
            }
            if (j + 1 < r) {
                edges.push_back({id, i * r + (j + 1)});
            }
        }
    }

    for (int i = (int)edges.size() - 1; i > 0; --i) {
        int j = rnd.next(i + 1);
        std::swap(edges[i], edges[j]);
    }

    DSU dsu(nodes);
    for (const auto &e : edges) {
        if (!dsu.unite(e.a, e.b)) {
            continue;
        }
        int ax = (e.a / r) * 2 + 1;
        int ay = (e.a % r) * 2 + 1;
        int bx = (e.b / r) * 2 + 1;
        int by = (e.b % r) * 2 + 1;
        int mx = (ax + bx) / 2;
        int my = (ay + by) / 2;
        grid[mx][my] = '.';
    }

    std::cout << N << " " << T << "\n";
    for (int i = 0; i < N; ++i) {
        std::cout << grid[i] << "\n";
    }
    return 0;
}
