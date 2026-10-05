#include "llama-ple-disk.h"

#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#endif

// The few file calls the reader makes, per platform: open (optionally uncached), a
// positional read that several threads may issue at once on the same handle, aligned
// bounce buffers, close. Everything above them is platform-neutral.
namespace {

#if defined(_WIN32)
using ple_fd = HANDLE;
static const ple_fd ple_fd_invalid = INVALID_HANDLE_VALUE;

static std::string ple_err_str(DWORD err) {
    LPSTR buf = nullptr;
    const DWORD size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &buf, 0, NULL);
    if (!size) {
        return format("Win32 error %lu", (unsigned long) err);
    }
    std::string ret(buf, size);
    LocalFree(buf);
    while (!ret.empty() && (ret.back() == '\n' || ret.back() == '\r' || ret.back() == ' ')) {
        ret.pop_back();
    }
    return ret;
}

static std::string ple_last_err() {
    return ple_err_str(GetLastError());
}

// FILE_FLAG_OVERLAPPED: a synchronous handle serializes every ReadFile on the file object,
// which would turn the reader pool into a queue of one. FILE_FLAG_NO_BUFFERING is the
// O_DIRECT analogue, with the same sector alignment rules the 4096-byte block logic obeys.
static ple_fd ple_open(const std::string & fname, bool direct) {
    const int wlen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, fname.c_str(), -1, nullptr, 0);
    if (wlen <= 0) {
        return ple_fd_invalid;
    }
    std::wstring wname(wlen, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, fname.c_str(), -1, &wname[0], wlen) <= 0) {
        return ple_fd_invalid;
    }
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS |
                        (direct ? FILE_FLAG_NO_BUFFERING : 0);
    return CreateFileW(wname.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
}

static void ple_close(ple_fd fd) {
    CloseHandle(fd);
}

// one manual-reset event per thread, for the OVERLAPPED of every read it issues
struct ple_thread_event {
    HANDLE h = NULL;
    ~ple_thread_event() {
        if (h) {
            CloseHandle(h);
        }
    }
};

// read up to len bytes at off; returns the count (0 at EOF), or -1 with err set
static int64_t ple_pread(ple_fd fd, uint8_t * dst, size_t len, int64_t off, std::string & err) {
    static thread_local ple_thread_event ev;
    if (ev.h == NULL) {
        ev.h = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (ev.h == NULL) {
            err = "CreateEvent: " + ple_last_err();
            return -1;
        }
    }

    OVERLAPPED ov = {};
    ov.Offset     = (DWORD) ((uint64_t) off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32);
    ov.hEvent     = ev.h;

    // ReadFile takes a DWORD; the reader asks for a row plus a block or two at most
    const DWORD want = (DWORD) std::min<size_t>(len, 1u << 30);
    DWORD       got  = 0;
    if (!ReadFile(fd, dst, want, NULL, &ov)) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) {
            return 0;
        }
        if (e != ERROR_IO_PENDING) {
            err = ple_err_str(e);
            return -1;
        }
    }
    if (!GetOverlappedResult(fd, &ov, &got, TRUE)) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) {
            return 0;
        }
        err = ple_err_str(e);
        return -1;
    }
    return (int64_t) got;
}

static void * ple_aligned_alloc(size_t align, size_t size) {
    return _aligned_malloc(size, align);
}

static void ple_aligned_free(void * ptr) {
    _aligned_free(ptr);
}
#else
using ple_fd = int;
static const ple_fd ple_fd_invalid = -1;

static std::string ple_last_err() {
    return strerror(errno);
}

static ple_fd ple_open(const std::string & fname, bool direct) {
    return open(fname.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
}

static void ple_close(ple_fd fd) {
    close(fd);
}

// read up to len bytes at off; returns the count (0 at EOF), or -1 with err set
static int64_t ple_pread(ple_fd fd, uint8_t * dst, size_t len, int64_t off, std::string & err) {
    for (;;) {
        const ssize_t r = pread(fd, dst, len, (off_t) off);
        if (r >= 0) {
            return (int64_t) r;
        }
        if (errno != EINTR) {
            err = strerror(errno);
            return -1;
        }
    }
}

static void * ple_aligned_alloc(size_t align, size_t size) {
    void * ptr = nullptr;
    return posix_memalign(&ptr, align, size) == 0 ? ptr : nullptr;
}

static void ple_aligned_free(void * ptr) {
    free(ptr);
}
#endif

} // namespace

struct llama_ple_disk::impl {
    std::string fname;
    ple_fd      fd     = ple_fd_invalid;
    bool        direct = false;
    size_t      offs   = 0;
    ggml_type   type   = GGML_TYPE_COUNT;
    int64_t     ne0    = 0;
    int64_t     nrows  = 0;
    size_t      rs     = 0;     // bytes per row
    size_t      block  = 4096;  // direct-I/O alignment for offset, length and buffer
    ggml_to_float_t to_float = nullptr;

    // direct-mapped cache of raw rows: slot = row & mask, tag = row
    size_t               n_slots = 0;
    size_t               mask    = 0;
    std::vector<int64_t> tags;
    std::vector<uint8_t> slab;

    // per-gather scratch, reused
    std::vector<int32_t>                       uniq;
    std::vector<uint8_t>                       raw;      // [uniq.size(), rs]
    std::vector<std::pair<int32_t, uint32_t>>  misses;   // (row, index into uniq)
    uint8_t *                                  bounce0 = nullptr; // main-thread bounce buffer

    std::mutex mtx; // one gather at a time; a model is shared by every context built on it

    // reader pool, started on first use
    int32_t                  n_threads = 1;
    std::vector<std::thread> workers;
    std::mutex               pm;
    std::condition_variable  cv_work;
    std::condition_variable  cv_done;
    uint64_t                 gen     = 0;
    size_t                   pending = 0;
    std::atomic<size_t>      next{0};
    bool                     stop    = false;

    uint64_t st_calls = 0, st_rows = 0, st_uniq = 0, st_hits = 0, st_reads = 0, st_bytes = 0;
    double   st_ms = 0;

    impl(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p)
        : fname(fname), offs(offs), type(type), ne0(ne0), nrows(nrows) {
        if (ggml_is_quantized(type) && ne0 % ggml_blck_size(type) != 0) {
            throw std::runtime_error(format("llama_ple_disk: row of %lld %s elements is not a whole number of blocks",
                                            (long long) ne0, ggml_type_name(type)));
        }
        rs = ggml_row_size(type, ne0);
        if (type != GGML_TYPE_F32) {
            to_float = ggml_get_type_traits(type)->to_float;
            if (to_float == nullptr) {
                throw std::runtime_error(format("llama_ple_disk: no dequantizer for %s", ggml_type_name(type)));
            }
        }

        direct = p.direct_io;
        if (direct) {
            fd = ple_open(fname, true);
            if (fd == ple_fd_invalid) {
                LLAMA_LOG_WARN("%s: direct-I/O open of %s failed (%s); falling back to buffered reads\n",
                               __func__, fname.c_str(), ple_last_err().c_str());
                direct = false;
            }
        }
        if (fd == ple_fd_invalid) {
            fd = ple_open(fname, false);
            if (fd == ple_fd_invalid) {
                throw std::runtime_error(format("llama_ple_disk: failed to open %s: %s", fname.c_str(), ple_last_err().c_str()));
            }
        }

        n_threads = std::max<int32_t>(1, p.n_threads);

        if (p.cache_bytes >= rs) {
            size_t n = p.cache_bytes / rs;
            n_slots = 1;
            while (n_slots * 2 <= n) {
                n_slots *= 2;
            }
            mask = n_slots - 1;
            tags.assign(n_slots, -1);
            slab.resize(n_slots * rs);
        }
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lk(pm);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            w.join();
        }
        ple_aligned_free(bounce0);
        if (fd != ple_fd_invalid) {
            ple_close(fd);
        }
    }

    size_t bounce_size() const {
        return ((rs + block - 1) / block) * block + 2 * block;
    }

    uint8_t * alloc_bounce() const {
        void * ptr = ple_aligned_alloc(block, bounce_size());
        if (ptr == nullptr) {
            throw std::runtime_error("llama_ple_disk: aligned allocation failed");
        }
        return (uint8_t *) ptr;
    }

    // read `len` bytes at `off` into `dst`; a short read is only tolerated past `need`
    void pread_full(uint8_t * dst, size_t len, int64_t off, size_t need) const {
        size_t got = 0;
        while (got < len) {
            std::string   err;
            const int64_t r = ple_pread(fd, dst + got, len - got, off + (int64_t) got, err);
            if (r < 0) {
                GGML_ABORT("llama_ple_disk: read(%s, %zu @ %lld) failed: %s",
                           fname.c_str(), len, (long long) off, err.c_str());
            }
            if (r == 0) {
                break; // EOF
            }
            got += (size_t) r;
        }
        if (got < need) {
            GGML_ABORT("llama_ple_disk: short read in %s: %zu of %zu bytes at %lld",
                       fname.c_str(), got, need, (long long) off);
        }
    }

    void read_row(int64_t row, uint8_t * dst, uint8_t * bounce) const {
        const int64_t off = (int64_t) offs + row * (int64_t) rs;
        if (!direct) {
            pread_full(dst, rs, off, rs);
            return;
        }
        const int64_t a0   = off & ~(int64_t) (block - 1);
        const size_t  need = (size_t) (off - a0) + rs;
        const size_t  len  = ((need + block - 1) / block) * block;
        pread_full(bounce, len, a0, need);
        memcpy(dst, bounce + (off - a0), rs);
    }

    void worker() {
        uint8_t * bounce = direct ? alloc_bounce() : nullptr;
        uint64_t  seen   = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pm);
                cv_work.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    break;
                }
                seen = gen;
            }
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= misses.size()) {
                    break;
                }
                read_row(misses[i].first, raw.data() + (size_t) misses[i].second * rs, bounce);
            }
            {
                std::lock_guard<std::mutex> lk(pm);
                if (--pending == 0) {
                    cv_done.notify_one();
                }
            }
        }
        ple_aligned_free(bounce);
    }

    void run_misses() {
        if (n_threads <= 1 || misses.size() <= 2) {
            if (direct && bounce0 == nullptr) {
                bounce0 = alloc_bounce();
            }
            for (const auto & m : misses) {
                read_row(m.first, raw.data() + (size_t) m.second * rs, bounce0);
            }
            return;
        }
        if (workers.empty()) {
            workers.reserve(n_threads);
            for (int32_t i = 0; i < n_threads; ++i) {
                workers.emplace_back([this] { worker(); });
            }
        }
        {
            std::lock_guard<std::mutex> lk(pm);
            next    = 0;
            pending = workers.size();
            ++gen;
        }
        cv_work.notify_all();
        std::unique_lock<std::mutex> lk(pm);
        cv_done.wait(lk, [&] { return pending == 0; });
    }

    void gather(const int32_t * idx, size_t n, float * dst) {
        std::lock_guard<std::mutex> lk(mtx);
        const auto t0 = std::chrono::steady_clock::now();

        uniq.assign(idx, idx + n);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (!uniq.empty() && (uniq.front() < 0 || (int64_t) uniq.back() >= nrows)) {
            GGML_ABORT("llama_ple_disk: row index out of range (%d..%d of %lld rows)",
                       uniq.front(), uniq.back(), (long long) nrows);
        }

        raw.resize(uniq.size() * rs);
        misses.clear();
        for (size_t i = 0; i < uniq.size(); ++i) {
            if (n_slots) {
                const size_t slot = (size_t) uniq[i] & mask;
                if (tags[slot] == uniq[i]) {
                    memcpy(raw.data() + i * rs, slab.data() + slot * rs, rs);
                    continue;
                }
            }
            misses.emplace_back(uniq[i], (uint32_t) i);
        }

        if (!misses.empty()) {
            run_misses();
            if (n_slots) {
                for (const auto & m : misses) {
                    const size_t slot = (size_t) m.first & mask;
                    tags[slot] = m.first;
                    memcpy(slab.data() + slot * rs, raw.data() + (size_t) m.second * rs, rs);
                }
            }
        }

        for (size_t k = 0; k < n; ++k) {
            const size_t    i   = (size_t) (std::lower_bound(uniq.begin(), uniq.end(), idx[k]) - uniq.begin());
            const uint8_t * src = raw.data() + i * rs;
            float *         out = dst + k * (size_t) ne0;
            if (to_float) {
                to_float(src, out, ne0);
            } else {
                memcpy(out, src, rs);
            }
        }

        st_calls += 1;
        st_rows  += n;
        st_uniq  += uniq.size();
        st_hits  += uniq.size() - misses.size();
        st_reads += misses.size();
        st_bytes += misses.size() * (direct ? block : rs);
        st_ms    += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

llama_ple_disk::llama_ple_disk(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p)
    : pimpl(std::make_unique<impl>(fname, offs, type, ne0, nrows, p)) {}

llama_ple_disk::~llama_ple_disk() {
    if (pimpl->st_calls) {
        LLAMA_LOG_INFO("%s: %llu batches, %llu rows gathered (%llu unique): %llu cache hits, %llu disk reads (%.1f MiB), %.1f ms total\n",
                       __func__, (unsigned long long) pimpl->st_calls, (unsigned long long) pimpl->st_rows,
                       (unsigned long long) pimpl->st_uniq, (unsigned long long) pimpl->st_hits,
                       (unsigned long long) pimpl->st_reads, pimpl->st_bytes / (1024.0 * 1024.0), pimpl->st_ms);
    }
}

void llama_ple_disk::gather(const int32_t * idx, size_t n, float * dst) {
    pimpl->gather(idx, n, dst);
}

int64_t llama_ple_disk::n_rows()   const { return pimpl->nrows; }
int64_t llama_ple_disk::ne0()      const { return pimpl->ne0;   }
size_t  llama_ple_disk::row_size() const { return pimpl->rs;    }

std::string llama_ple_disk::describe() const {
    return format("%s @ %zu: %lld rows x %lld %s (%zu B/row, %.2f GiB), %s I/O, %d threads, row cache %zu MiB",
                  pimpl->fname.c_str(), pimpl->offs, (long long) pimpl->nrows, (long long) pimpl->ne0,
                  ggml_type_name(pimpl->type), pimpl->rs, (double) pimpl->nrows * pimpl->rs / (1024.0 * 1024.0 * 1024.0),
                  pimpl->direct ? "direct" : "buffered", pimpl->n_threads,
                  pimpl->n_slots * pimpl->rs / (1024 * 1024));
}
