#include "fft_processor_concrete.hpp"

// FFT_Processor_MKL is thread-safe
thread_local FFT_Processor_Concrete<TFHEpp::lvl1param::n> fftplvl1;
thread_local FFT_Processor_Concrete<TFHEpp::lvl2param::n> fftplvl2;