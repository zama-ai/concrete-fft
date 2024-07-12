#pragma once

#include <cxx.hpp>
#include <concrete-fft.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>

#include<params.hpp>
#define CAST_DOUBLE_TO_UINT32(d) ((uint32_t)((int64_t)(d)))

template<uint N>
class FFT_Processor_Concrete {
private:
    static constexpr uint Ns2 = N / 2;
    static constexpr double _2sN = double(2) / double(N);
    alignas(64) std::array<std::complex<double>,Ns2> twist;
    alignas(64) std::vector<uint8_t> scratch_memory;
    rust::Slice<uint8_t> scratch_slice;


public:
    FFT_Processor_Concrete(){
        for (int i = 0; i < Ns2; i++) {
            double value = (double)i * M_PI / (double)N;
            twist[i] = (std::complex<double>(std::cos(value), std::sin(value)));
        }
        // scratch_memory.resize(concrete_fft::stacksize(N)+32);
        scratch_memory.resize(N*16);
        scratch_slice = rust::Slice<uint8_t>(scratch_memory.data(), scratch_memory.size());
    }

    void execute_reverse_int(double *res, const int32_t *a)
    {
        for (int i = 0; i < Ns2; i++) {
            auto tmp = twist[i] * std::complex((double)a[i], (double)a[Ns2 + i]);
            res[2*i] = tmp.real();
            res[2*i+1] = tmp.imag();
        }
        rust::Slice<double> res_slice(res, N);
        concrete_fft::fwd(res_slice, scratch_slice);
    }

    void execute_reverse_torus32(double *res, const uint32_t *a)
    {
        execute_reverse_int(res, (int32_t *)a);
    }

    void execute_direct_torus32(uint32_t *res, const double *a)
    {
        std::array<double, N> buf;
        for(int i = 0; i < N; i++) buf[i] = a[i];
        rust::Slice<double> buf_slice(buf.data(), N);
        concrete_fft::inv(buf_slice, scratch_slice);
        for (int i = 0; i < Ns2; i++) {
            auto res_tmp =
                std::complex<double>(buf[2*i], buf[2*i+1]) * std::conj(twist[i]) * _2sN;
            res[i] = CAST_DOUBLE_TO_UINT32(res_tmp.real());
            res[i + Ns2] = CAST_DOUBLE_TO_UINT32(res_tmp.imag());
        }
    }

    void execute_direct_torus32_rescale(uint32_t *res, const double *a,
                                        const double Δ)
    {
        std::array<double, N> buf;
        for(int i = 0; i < N; i++) buf[i] = a[i];
        rust::Slice<double> buf_slice(buf.data(), N);
        concrete_fft::inv(buf_slice, scratch_slice);
        for (int i = 0; i < Ns2; i++) {
            auto res_tmp =
                std::complex<double>(buf[2*i], buf[2*i+1]) * std::conj(twist[i]) * _2sN;
            res[i] = CAST_DOUBLE_TO_UINT32(res_tmp.real() / (Δ / 4));
            res[i + Ns2] = CAST_DOUBLE_TO_UINT32(res_tmp.imag() / (Δ / 4));
        }
    }

    void execute_reverse_torus64(double *res, const uint64_t *a)
    {
        for (int i = 0; i < Ns2; i++) {
            auto tmp = twist[i] * std::complex((double)((int64_t)a[i]),
                                            (double)((int64_t)a[Ns2 + i]));
            res[2*i] = tmp.real();
            res[2*i+1] = tmp.imag();
        }
        rust::Slice<double> res_slice(res, N);
        concrete_fft::fwd(res_slice, scratch_slice);
    }

    void execute_direct_torus64(uint64_t *res, const double *a)
    {
        std::array<double, N> buf;
        for(int i = 0; i < N; i++) buf[i] = a[i];
        rust::Slice<double> buf_slice(buf.data(), N);
        concrete_fft::inv(buf_slice, scratch_slice);
        std::array<double, N> tmp;
        for (int i = 0; i < Ns2; i++) {
            auto res_tmp =
                std::complex<double>(buf[2*i], buf[2*i+1]) * std::conj(twist[i]) * _2sN;
            tmp[i] = res_tmp.real();
            tmp[i + Ns2] = res_tmp.imag();
        }
        const uint64_t *const vals = (const uint64_t *)tmp.data();
        constexpr uint64_t valmask0 = 0x000FFFFFFFFFFFFFul;
        constexpr uint64_t valmask1 = 0x0010000000000000ul;
        constexpr uint16_t expmask0 = 0x07FFu;
        for (int i = 0; i < N; i++) {
            uint64_t val = (vals[i] & valmask0) | valmask1;  // mantissa on 53 bits
            uint16_t expo = (vals[i] >> 52) & expmask0;      // exponent 11 bits
            // 1023 -> 52th pos -> 0th pos
            // 1075 -> 52th pos -> 52th pos
            int16_t trans = expo - 1075;
            uint64_t val2 = trans > 0 ? (val << trans) : (val >> -trans);
            res[i] = (vals[i] >> 63) ? -val2 : val2;
        }
    }

    void execute_direct_torus64_rescale(uint64_t *res, const double *a, const double Δ)
    {
        std::array<double, N> buf;
        for(int i = 0; i < N; i++) buf[i] = a[i];
        rust::Slice<double> buf_slice(buf.data(), N);
        concrete_fft::inv(buf_slice, scratch_slice);
        std::array<double,N> tmp;
        for (int i = 0; i < Ns2; i++) {
            auto res_tmp =
                std::complex<double>(buf[2*i], buf[2*i+1]) * std::conj(twist[i]) * _2sN;
            tmp[i] = res_tmp.real();
            tmp[i + Ns2] = res_tmp.imag();
        }
        for (int i=0; i<N; i++) res[i] = uint64_t(std::round(tmp[i]/(Δ/4)));
    }

    // ~FFT_Processor_Concrete();
};

#undef CAST_DOUBLE_TO_UINT32

// FFT_Processor_Concrete is thread-safe
extern thread_local FFT_Processor_Concrete<TFHEpp::lvl1param::n> fftplvl1;
extern thread_local FFT_Processor_Concrete<TFHEpp::lvl2param::n> fftplvl2;