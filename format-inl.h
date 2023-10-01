// Formatting library for C++ - implementation
//
// Copyright (c) 2012 - 2016, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_FORMAT_INL_H_
#define FMT_FORMAT_INL_H_

#include <algorithm>
#include <cerrno>  // errno
#include <climits>
#include <cmath>
#include <exception>

#ifndef FMT_STATIC_THOUSANDS_SEPARATOR
#  include <locale>
#endif

#if defined(_WIN32) && !defined(FMT_WINDOWS_NO_WCHAR)
#  include <io.h>  // _isatty
#endif

#include "format.h"

FMT_BEGIN_NAMESPACE
namespace detail {

FMT_FUNC void assert_fail(const char* file, int line, const char* message) {
  // Use unchecked std::fprintf to avoid triggering another assertion when
  // writing to stderr fails
  std::fprintf(stderr, "%s:%d: assertion failed: %s", file, line, message);
  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
  // code pass.
  std::terminate();
}

FMT_FUNC void throw_format_error(const char* message) {
  FMT_THROW(format_error(message));
}

FMT_FUNC void format_error_code(detail::buffer<char>& out, int error_code,
                                string_view message) noexcept {
  // Report error code making sure that the output fits into
  // inline_buffer_size to avoid dynamic memory allocation and potential
  // bad_alloc.
  out.try_resize(0);
  static const char SEP[] = ": ";
  static const char ERROR_STR[] = "error ";
  // Subtract 2 to account for terminating null characters in SEP and ERROR_STR.
  size_t error_code_size = sizeof(SEP) + sizeof(ERROR_STR) - 2;
  auto abs_value = static_cast<uint32_or_64_or_128_t<int>>(error_code);
  if (detail::is_negative(error_code)) {
    abs_value = 0 - abs_value;
    ++error_code_size;
  }
  error_code_size += detail::to_unsigned(detail::count_digits(abs_value));
  auto it = buffer_appender<char>(out);
  if (message.size() <= inline_buffer_size - error_code_size)
    format_to(it, FMT_STRING("{}{}"), message, SEP);
  format_to(it, FMT_STRING("{}{}"), ERROR_STR, error_code);
  FMT_ASSERT(out.size() <= inline_buffer_size, "");
}

FMT_FUNC void report_error(format_func func, int error_code,
                           const char* message) noexcept {
  memory_buffer full_message;
  func(full_message, error_code, message);
  // Don't use fwrite_fully because the latter may throw.
  if (std::fwrite(full_message.data(), full_message.size(), 1, stderr) > 0)
    std::fputc('\n', stderr);
}

// A wrapper around fwrite that throws on error.
inline void fwrite_fully(const void* ptr, size_t size, size_t count,
                         FILE* stream) {
  size_t written = std::fwrite(ptr, size, count, stream);
  if (written < count)
    FMT_THROW(system_error(errno, FMT_STRING("cannot write to file")));
}

#ifndef FMT_STATIC_THOUSANDS_SEPARATOR
template <typename Locale>
locale_ref::locale_ref(const Locale& loc) : locale_(&loc) {
  static_assert(std::is_same<Locale, std::locale>::value, "");
}

//TODO Local
template <typename Locale> Locale locale_ref::get() const {
  static_assert(std::is_same<Locale, std::locale>::value, "");
  return locale_ ? *static_cast<const std::locale*>(locale_) : std::locale();
}

//TODO Facet
template <typename Char>
FMT_FUNC auto thousands_sep_impl(locale_ref loc) -> thousands_sep_result<Char> {
  auto& facet = std::use_facet<std::numpunct<Char>>(loc.get<std::locale>());
  auto grouping = facet.grouping();
  auto thousands_sep = grouping.empty() ? Char() : facet.thousands_sep();
  return thousands_sep_result<Char>(std::move(grouping), thousands_sep);
}
template <typename Char> FMT_FUNC Char decimal_point_impl(locale_ref loc) {
  return std::use_facet<std::numpunct<Char>>(loc.get<std::locale>())
      .decimal_point();
}
#else
template <typename Char>
FMT_FUNC auto thousands_sep_impl(locale_ref) -> thousands_sep_result<Char> {
  return {"\03", FMT_STATIC_THOUSANDS_SEPARATOR};
}
template <typename Char> FMT_FUNC Char decimal_point_impl(locale_ref) {
  return '.';
}
#endif

FMT_FUNC auto write_loc(appender out, loc_value value,
                        const format_specs<>& specs, locale_ref loc) -> bool {
#ifndef FMT_STATIC_THOUSANDS_SEPARATOR
  auto locale = loc.get<std::locale>();
  // We cannot use the num_put<char> facet because it may produce output in
  // a wrong encoding.
  using facet = format_facet<std::locale>;
  if (std::has_facet<facet>(locale))
    return std::use_facet<facet>(locale).put(out, value, specs);
  return facet(locale).put(out, value, specs);
#endif
  return false;
}
}  // namespace detail

template <typename Locale> typename Locale::id format_facet<Locale>::id;

#ifndef FMT_STATIC_THOUSANDS_SEPARATOR
template <typename Locale> format_facet<Locale>::format_facet(Locale& loc) {
  auto& numpunct = std::use_facet<std::numpunct<char>>(loc);
  grouping_ = numpunct.grouping();
  if (!grouping_.empty()) separator_ = std::string(1, numpunct.thousands_sep());
}

template <>
FMT_API FMT_FUNC auto format_facet<std::locale>::do_put(
    appender out, loc_value val, const format_specs<>& specs) const -> bool {
  return val.visit(
      detail::loc_writer<>{out, specs, separator_, grouping_, decimal_point_});
}
#endif

FMT_FUNC std::system_error vsystem_error(int error_code, string_view fmt,
                                         format_args args) {
  auto ec = std::error_code(error_code, std::generic_category());
  return std::system_error(ec, vformat(fmt, args));
}

namespace detail {

template <typename F> inline bool operator==(basic_fp<F> x, basic_fp<F> y) {
  return x.f == y.f && x.e == y.e;
}

// Compilers should be able to optimize this into the ror instruction.
FMT_CONSTEXPR inline uint32_t rotr(uint32_t n, uint32_t r) noexcept {
  r &= 31;
  return (n >> r) | (n << (32 - r));
}
FMT_CONSTEXPR inline uint64_t rotr(uint64_t n, uint32_t r) noexcept {
  r &= 63;
  return (n >> r) | (n << (64 - r));
}

// Implementation of Dragonbox algorithm: https://github.com/jk-jeon/dragonbox.
namespace dragonbox {
// Computes upper 64 bits of multiplication of a 32-bit unsigned integer and a
// 64-bit unsigned integer.
inline uint64_t umul96_upper64(uint32_t x, uint64_t y) noexcept {
  return umul128_upper64(static_cast<uint64_t>(x) << 32, y);
}

// Computes lower 128 bits of multiplication of a 64-bit unsigned integer and a
// 128-bit unsigned integer.
inline uint128_fallback umul192_lower128(uint64_t x,
                                         uint128_fallback y) noexcept {
  uint64_t high = x * y.high();
  uint128_fallback high_low = umul128(x, y.low());
  return {high + high_low.high(), high_low.low()};
}

// Computes lower 64 bits of multiplication of a 32-bit unsigned integer and a
// 64-bit unsigned integer.
inline uint64_t umul96_lower64(uint32_t x, uint64_t y) noexcept {
  return x * y;
}

// Various fast log computations.
inline int floor_log10_pow2_minus_log10_4_over_3(int e) noexcept {
  FMT_ASSERT(e <= 2936 && e >= -2985, "too large exponent");
  return (e * 631305 - 261663) >> 21;
}

FMT_INLINE_VARIABLE constexpr struct {
  uint32_t divisor;
  int shift_amount;
} div_small_pow10_infos[] = {{10, 16}, {100, 16}};

// Replaces n by floor(n / pow(10, N)) returning true if and only if n is
// divisible by pow(10, N).
// Precondition: n <= pow(10, N + 1).
template <int N>
bool check_divisibility_and_divide_by_pow10(uint32_t& n) noexcept {
  // The numbers below are chosen such that:
  //   1. floor(n/d) = floor(nm / 2^k) where d=10 or d=100,
  //   2. nm mod 2^k < m if and only if n is divisible by d,
  // where m is magic_number, k is shift_amount
  // and d is divisor.
  //
  // Item 1 is a common technique of replacing division by a constant with
  // multiplication, see e.g. "Division by Invariant Integers Using
  // Multiplication" by Granlund and Montgomery (1994). magic_number (m) is set
  // to ceil(2^k/d) for large enough k.
  // The idea for item 2 originates from Schubfach.
  constexpr auto info = div_small_pow10_infos[N - 1];
  FMT_ASSERT(n <= info.divisor * 10, "n is too large");
  constexpr uint32_t magic_number =
      (1u << info.shift_amount) / info.divisor + 1;
  n *= magic_number;
  const uint32_t comparison_mask = (1u << info.shift_amount) - 1;
  bool result = (n & comparison_mask) < magic_number;
  n >>= info.shift_amount;
  return result;
}

// Computes floor(n / pow(10, N)) for small n and N.
// Precondition: n <= pow(10, N + 1).
template <int N> uint32_t small_division_by_pow10(uint32_t n) noexcept {
  constexpr auto info = div_small_pow10_infos[N - 1];
  FMT_ASSERT(n <= info.divisor * 10, "n is too large");
  constexpr uint32_t magic_number =
      (1u << info.shift_amount) / info.divisor + 1;
  return (n * magic_number) >> info.shift_amount;
}

// Computes floor(n / 10^(kappa + 1)) (float)
inline uint32_t divide_by_10_to_kappa_plus_1(uint32_t n) noexcept {
  // 1374389535 = ceil(2^37/100)
  return static_cast<uint32_t>((static_cast<uint64_t>(n) * 1374389535) >> 37);
}
// Computes floor(n / 10^(kappa + 1)) (double)
inline uint64_t divide_by_10_to_kappa_plus_1(uint64_t n) noexcept {
  // 2361183241434822607 = ceil(2^(64+7)/1000)
  return umul128_upper64(n, 2361183241434822607ull) >> 7;
}

// Various integer checks
template <typename T>
bool is_left_endpoint_integer_shorter_interval(int exponent) noexcept {
  const int case_shorter_interval_left_endpoint_lower_threshold = 2;
  const int case_shorter_interval_left_endpoint_upper_threshold = 3;
  return exponent >= case_shorter_interval_left_endpoint_lower_threshold &&
         exponent <= case_shorter_interval_left_endpoint_upper_threshold;
}

// Remove trailing zeros from n and return the number of zeros removed (float)
FMT_INLINE int remove_trailing_zeros(uint32_t& n, int s = 0) noexcept {
  FMT_ASSERT(n != 0, "");
  // Modular inverse of 5 (mod 2^32): (mod_inv_5 * 5) mod 2^32 = 1.
  constexpr uint32_t mod_inv_5 = 0xcccccccd;
  constexpr uint32_t mod_inv_25 = 0xc28f5c29; // = mod_inv_5 * mod_inv_5

  while (true) {
    auto q = rotr(n * mod_inv_25, 2);
    if (q > max_value<uint32_t>() / 100) break;
    n = q;
    s += 2;
  }
  auto q = rotr(n * mod_inv_5, 1);
  if (q <= max_value<uint32_t>() / 10) {
    n = q;
    s |= 1;
  }
  return s;
}

// Removes trailing zeros and returns the number of zeros removed (double)
FMT_INLINE int remove_trailing_zeros(uint64_t& n) noexcept {
  FMT_ASSERT(n != 0, "");

  // This magic number is ceil(2^90 / 10^8).
  constexpr uint64_t magic_number = 12379400392853802749ull;
  auto nm = umul128(n, magic_number);

  // Is n is divisible by 10^8?
  if ((nm.high() & ((1ull << (90 - 64)) - 1)) == 0 && nm.low() < magic_number) {
    // If yes, work with the quotient...
    auto n32 = static_cast<uint32_t>(nm.high() >> (90 - 64));
    // ... and use the 32 bit variant of the function
    int s = remove_trailing_zeros(n32, 8);
    n = n32;
    return s;
  }

  // If n is not divisible by 10^8, work with n itself.
  constexpr uint64_t mod_inv_5 = 0xcccccccccccccccd;
  constexpr uint64_t mod_inv_25 = 0x8f5c28f5c28f5c29; // = mod_inv_5 * mod_inv_5

  int s = 0;
  while (true) {
    auto q = rotr(n * mod_inv_25, 2);
    if (q > max_value<uint64_t>() / 100) break;
    n = q;
    s += 2;
  }
  auto q = rotr(n * mod_inv_5, 1);
  if (q <= max_value<uint64_t>() / 10) {
    n = q;
    s |= 1;
  }

  return s;
}
}  // namespace dragonbox
}  // namespace detail

template <> struct formatter<detail::bigint> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx)
      -> format_parse_context::iterator {
    return ctx.begin();
  }

  auto format(const detail::bigint& n, format_context& ctx) const
      -> format_context::iterator {
    auto out = ctx.out();
    bool first = true;
    for (auto i = n.bigits_.size(); i > 0; --i) {
      auto value = n.bigits_[i - 1u];
      if (first) {
        out = format_to(out, FMT_STRING("{:x}"), value);
        first = false;
        continue;
      }
      out = format_to(out, FMT_STRING("{:08x}"), value);
    }
    if (n.exp_ > 0)
      out = format_to(out, FMT_STRING("p{}"),
                      n.exp_ * detail::bigint::bigit_bits);
    return out;
  }
};

FMT_FUNC detail::utf8_to_utf16::utf8_to_utf16(string_view s) {
  for_each_codepoint(s, [this](uint32_t cp, string_view) {
    if (cp == invalid_code_point) FMT_THROW(std::runtime_error("invalid utf8"));
    if (cp <= 0xFFFF) {
      buffer_.push_back(static_cast<wchar_t>(cp));
    } else {
      cp -= 0x10000;
      buffer_.push_back(static_cast<wchar_t>(0xD800 + (cp >> 10)));
      buffer_.push_back(static_cast<wchar_t>(0xDC00 + (cp & 0x3FF)));
    }
    return true;
  });
  buffer_.push_back(0);
}

FMT_FUNC void format_system_error(detail::buffer<char>& out, int error_code,
                                  const char* message) noexcept {
  FMT_TRY {
    auto ec = std::error_code(error_code, std::generic_category());
    write(std::back_inserter(out), std::system_error(ec, message).what());
    return;
  }
  FMT_CATCH(...) {}
  format_error_code(out, error_code, message);
}

FMT_FUNC void report_system_error(int error_code,
                                  const char* message) noexcept {
  report_error(format_system_error, error_code, message);
}

FMT_FUNC std::string vformat(string_view fmt, format_args args) {
  // Don't optimize the "{}" case to keep the binary size small and because it
  // can be better optimized in fmt::format anyway.
  auto buffer = memory_buffer();
  detail::vformat_to(buffer, fmt, args);
  return to_string(buffer);
}

namespace detail {
#if !defined(_WIN32) || defined(FMT_WINDOWS_NO_WCHAR)
FMT_FUNC bool write_console(std::FILE*, string_view) { return false; }
#else
using dword = conditional_t<sizeof(long) == 4, unsigned long, unsigned>;
extern "C" __declspec(dllimport) int __stdcall WriteConsoleW(  //
    void*, const void*, dword, dword*, void*);

FMT_FUNC bool write_console(std::FILE* f, string_view text) {
  auto fd = _fileno(f);
  if (!_isatty(fd)) return false;
  auto u16 = utf8_to_utf16(text);
  auto written = dword();
  return WriteConsoleW(reinterpret_cast<void*>(_get_osfhandle(fd)), u16.c_str(),
                       static_cast<uint32_t>(u16.size()), &written, nullptr) != 0;
}
#endif

#ifdef _WIN32
// Print assuming legacy (non-Unicode) encoding.
FMT_FUNC void vprint_mojibake(std::FILE* f, string_view fmt, format_args args) {
  auto buffer = memory_buffer();
  detail::vformat_to(buffer, fmt, args);
  fwrite_fully(buffer.data(), 1, buffer.size(), f);
}
#endif

FMT_FUNC void print(std::FILE* f, string_view text) {
  if (!write_console(f, text)) fwrite_fully(text.data(), 1, text.size(), f);
}
}  // namespace detail

FMT_FUNC void vprint(std::FILE* f, string_view fmt, format_args args) {
  auto buffer = memory_buffer();
  detail::vformat_to(buffer, fmt, args);
  detail::print(f, {buffer.data(), buffer.size()});
}

FMT_FUNC void vprint(string_view fmt, format_args args) {
  vprint(stdout, fmt, args);
}

namespace detail {

struct singleton {
  unsigned char upper;
  unsigned char lower_count;
};

inline auto is_printable(uint16_t x, const singleton* singletons,
                         size_t singletons_size,
                         const unsigned char* singleton_lowers,
                         const unsigned char* normal, size_t normal_size)
    -> bool {
  auto upper = x >> 8;
  auto lower_start = 0;
  for (size_t i = 0; i < singletons_size; ++i) {
    auto s = singletons[i];
    auto lower_end = lower_start + s.lower_count;
    if (upper < s.upper) break;
    if (upper == s.upper) {
      for (auto j = lower_start; j < lower_end; ++j) {
        if (singleton_lowers[j] == (x & 0xff)) return false;
      }
    }
    lower_start = lower_end;
  }

  auto xsigned = static_cast<int>(x);
  auto current = true;
  for (size_t i = 0; i < normal_size; ++i) {
    auto v = static_cast<int>(normal[i]);
    auto len = (v & 0x80) != 0 ? (v & 0x7f) << 8 | normal[++i] : v;
    xsigned -= len;
    if (xsigned < 0) break;
    current = !current;
  }
  return current;
}

// This code is generated by support/printable.py.
FMT_FUNC auto is_printable(uint32_t cp) -> bool {
  static constexpr singleton singletons0[] = {
      {0x00, 1},  {0x03, 5},  {0x05, 6},  {0x06, 3},  {0x07, 6},  {0x08, 8},
      {0x09, 17}, {0x0a, 28}, {0x0b, 25}, {0x0c, 20}, {0x0d, 16}, {0x0e, 13},
      {0x0f, 4},  {0x10, 3},  {0x12, 18}, {0x13, 9},  {0x16, 1},  {0x17, 5},
      {0x18, 2},  {0x19, 3},  {0x1a, 7},  {0x1c, 2},  {0x1d, 1},  {0x1f, 22},
      {0x20, 3},  {0x2b, 3},  {0x2c, 2},  {0x2d, 11}, {0x2e, 1},  {0x30, 3},
      {0x31, 2},  {0x32, 1},  {0xa7, 2},  {0xa9, 2},  {0xaa, 4},  {0xab, 8},
      {0xfa, 2},  {0xfb, 5},  {0xfd, 4},  {0xfe, 3},  {0xff, 9},
  };
  static constexpr unsigned char singletons0_lower[] = {
      0xad, 0x78, 0x79, 0x8b, 0x8d, 0xa2, 0x30, 0x57, 0x58, 0x8b, 0x8c, 0x90,
      0x1c, 0x1d, 0xdd, 0x0e, 0x0f, 0x4b, 0x4c, 0xfb, 0xfc, 0x2e, 0x2f, 0x3f,
      0x5c, 0x5d, 0x5f, 0xb5, 0xe2, 0x84, 0x8d, 0x8e, 0x91, 0x92, 0xa9, 0xb1,
      0xba, 0xbb, 0xc5, 0xc6, 0xc9, 0xca, 0xde, 0xe4, 0xe5, 0xff, 0x00, 0x04,
      0x11, 0x12, 0x29, 0x31, 0x34, 0x37, 0x3a, 0x3b, 0x3d, 0x49, 0x4a, 0x5d,
      0x84, 0x8e, 0x92, 0xa9, 0xb1, 0xb4, 0xba, 0xbb, 0xc6, 0xca, 0xce, 0xcf,
      0xe4, 0xe5, 0x00, 0x04, 0x0d, 0x0e, 0x11, 0x12, 0x29, 0x31, 0x34, 0x3a,
      0x3b, 0x45, 0x46, 0x49, 0x4a, 0x5e, 0x64, 0x65, 0x84, 0x91, 0x9b, 0x9d,
      0xc9, 0xce, 0xcf, 0x0d, 0x11, 0x29, 0x45, 0x49, 0x57, 0x64, 0x65, 0x8d,
      0x91, 0xa9, 0xb4, 0xba, 0xbb, 0xc5, 0xc9, 0xdf, 0xe4, 0xe5, 0xf0, 0x0d,
      0x11, 0x45, 0x49, 0x64, 0x65, 0x80, 0x84, 0xb2, 0xbc, 0xbe, 0xbf, 0xd5,
      0xd7, 0xf0, 0xf1, 0x83, 0x85, 0x8b, 0xa4, 0xa6, 0xbe, 0xbf, 0xc5, 0xc7,
      0xce, 0xcf, 0xda, 0xdb, 0x48, 0x98, 0xbd, 0xcd, 0xc6, 0xce, 0xcf, 0x49,
      0x4e, 0x4f, 0x57, 0x59, 0x5e, 0x5f, 0x89, 0x8e, 0x8f, 0xb1, 0xb6, 0xb7,
      0xbf, 0xc1, 0xc6, 0xc7, 0xd7, 0x11, 0x16, 0x17, 0x5b, 0x5c, 0xf6, 0xf7,
      0xfe, 0xff, 0x80, 0x0d, 0x6d, 0x71, 0xde, 0xdf, 0x0e, 0x0f, 0x1f, 0x6e,
      0x6f, 0x1c, 0x1d, 0x5f, 0x7d, 0x7e, 0xae, 0xaf, 0xbb, 0xbc, 0xfa, 0x16,
      0x17, 0x1e, 0x1f, 0x46, 0x47, 0x4e, 0x4f, 0x58, 0x5a, 0x5c, 0x5e, 0x7e,
      0x7f, 0xb5, 0xc5, 0xd4, 0xd5, 0xdc, 0xf0, 0xf1, 0xf5, 0x72, 0x73, 0x8f,
      0x74, 0x75, 0x96, 0x2f, 0x5f, 0x26, 0x2e, 0x2f, 0xa7, 0xaf, 0xb7, 0xbf,
      0xc7, 0xcf, 0xd7, 0xdf, 0x9a, 0x40, 0x97, 0x98, 0x30, 0x8f, 0x1f, 0xc0,
      0xc1, 0xce, 0xff, 0x4e, 0x4f, 0x5a, 0x5b, 0x07, 0x08, 0x0f, 0x10, 0x27,
      0x2f, 0xee, 0xef, 0x6e, 0x6f, 0x37, 0x3d, 0x3f, 0x42, 0x45, 0x90, 0x91,
      0xfe, 0xff, 0x53, 0x67, 0x75, 0xc8, 0xc9, 0xd0, 0xd1, 0xd8, 0xd9, 0xe7,
      0xfe, 0xff,
  };
  static constexpr singleton singletons1[] = {
      {0x00, 6},  {0x01, 1}, {0x03, 1},  {0x04, 2}, {0x08, 8},  {0x09, 2},
      {0x0a, 5},  {0x0b, 2}, {0x0e, 4},  {0x10, 1}, {0x11, 2},  {0x12, 5},
      {0x13, 17}, {0x14, 1}, {0x15, 2},  {0x17, 2}, {0x19, 13}, {0x1c, 5},
      {0x1d, 8},  {0x24, 1}, {0x6a, 3},  {0x6b, 2}, {0xbc, 2},  {0xd1, 2},
      {0xd4, 12}, {0xd5, 9}, {0xd6, 2},  {0xd7, 2}, {0xda, 1},  {0xe0, 5},
      {0xe1, 2},  {0xe8, 2}, {0xee, 32}, {0xf0, 4}, {0xf8, 2},  {0xf9, 2},
      {0xfa, 2},  {0xfb, 1},
  };
  static constexpr unsigned char singletons1_lower[] = {
      0x0c, 0x27, 0x3b, 0x3e, 0x4e, 0x4f, 0x8f, 0x9e, 0x9e, 0x9f, 0x06, 0x07,
      0x09, 0x36, 0x3d, 0x3e, 0x56, 0xf3, 0xd0, 0xd1, 0x04, 0x14, 0x18, 0x36,
      0x37, 0x56, 0x57, 0x7f, 0xaa, 0xae, 0xaf, 0xbd, 0x35, 0xe0, 0x12, 0x87,
      0x89, 0x8e, 0x9e, 0x04, 0x0d, 0x0e, 0x11, 0x12, 0x29, 0x31, 0x34, 0x3a,
      0x45, 0x46, 0x49, 0x4a, 0x4e, 0x4f, 0x64, 0x65, 0x5c, 0xb6, 0xb7, 0x1b,
      0x1c, 0x07, 0x08, 0x0a, 0x0b, 0x14, 0x17, 0x36, 0x39, 0x3a, 0xa8, 0xa9,
      0xd8, 0xd9, 0x09, 0x37, 0x90, 0x91, 0xa8, 0x07, 0x0a, 0x3b, 0x3e, 0x66,
      0x69, 0x8f, 0x92, 0x6f, 0x5f, 0xee, 0xef, 0x5a, 0x62, 0x9a, 0x9b, 0x27,
      0x28, 0x55, 0x9d, 0xa0, 0xa1, 0xa3, 0xa4, 0xa7, 0xa8, 0xad, 0xba, 0xbc,
      0xc4, 0x06, 0x0b, 0x0c, 0x15, 0x1d, 0x3a, 0x3f, 0x45, 0x51, 0xa6, 0xa7,
      0xcc, 0xcd, 0xa0, 0x07, 0x19, 0x1a, 0x22, 0x25, 0x3e, 0x3f, 0xc5, 0xc6,
      0x04, 0x20, 0x23, 0x25, 0x26, 0x28, 0x33, 0x38, 0x3a, 0x48, 0x4a, 0x4c,
      0x50, 0x53, 0x55, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x63, 0x65, 0x66,
      0x6b, 0x73, 0x78, 0x7d, 0x7f, 0x8a, 0xa4, 0xaa, 0xaf, 0xb0, 0xc0, 0xd0,
      0xae, 0xaf, 0x79, 0xcc, 0x6e, 0x6f, 0x93,
  };
  static constexpr unsigned char normal0[] = {
      0x00, 0x20, 0x5f, 0x22, 0x82, 0xdf, 0x04, 0x82, 0x44, 0x08, 0x1b, 0x04,
      0x06, 0x11, 0x81, 0xac, 0x0e, 0x80, 0xab, 0x35, 0x28, 0x0b, 0x80, 0xe0,
      0x03, 0x19, 0x08, 0x01, 0x04, 0x2f, 0x04, 0x34, 0x04, 0x07, 0x03, 0x01,
      0x07, 0x06, 0x07, 0x11, 0x0a, 0x50, 0x0f, 0x12, 0x07, 0x55, 0x07, 0x03,
      0x04, 0x1c, 0x0a, 0x09, 0x03, 0x08, 0x03, 0x07, 0x03, 0x02, 0x03, 0x03,
      0x03, 0x0c, 0x04, 0x05, 0x03, 0x0b, 0x06, 0x01, 0x0e, 0x15, 0x05, 0x3a,
      0x03, 0x11, 0x07, 0x06, 0x05, 0x10, 0x07, 0x57, 0x07, 0x02, 0x07, 0x15,
      0x0d, 0x50, 0x04, 0x43, 0x03, 0x2d, 0x03, 0x01, 0x04, 0x11, 0x06, 0x0f,
      0x0c, 0x3a, 0x04, 0x1d, 0x25, 0x5f, 0x20, 0x6d, 0x04, 0x6a, 0x25, 0x80,
      0xc8, 0x05, 0x82, 0xb0, 0x03, 0x1a, 0x06, 0x82, 0xfd, 0x03, 0x59, 0x07,
      0x15, 0x0b, 0x17, 0x09, 0x14, 0x0c, 0x14, 0x0c, 0x6a, 0x06, 0x0a, 0x06,
      0x1a, 0x06, 0x59, 0x07, 0x2b, 0x05, 0x46, 0x0a, 0x2c, 0x04, 0x0c, 0x04,
      0x01, 0x03, 0x31, 0x0b, 0x2c, 0x04, 0x1a, 0x06, 0x0b, 0x03, 0x80, 0xac,
      0x06, 0x0a, 0x06, 0x21, 0x3f, 0x4c, 0x04, 0x2d, 0x03, 0x74, 0x08, 0x3c,
      0x03, 0x0f, 0x03, 0x3c, 0x07, 0x38, 0x08, 0x2b, 0x05, 0x82, 0xff, 0x11,
      0x18, 0x08, 0x2f, 0x11, 0x2d, 0x03, 0x20, 0x10, 0x21, 0x0f, 0x80, 0x8c,
      0x04, 0x82, 0x97, 0x19, 0x0b, 0x15, 0x88, 0x94, 0x05, 0x2f, 0x05, 0x3b,
      0x07, 0x02, 0x0e, 0x18, 0x09, 0x80, 0xb3, 0x2d, 0x74, 0x0c, 0x80, 0xd6,
      0x1a, 0x0c, 0x05, 0x80, 0xff, 0x05, 0x80, 0xdf, 0x0c, 0xee, 0x0d, 0x03,
      0x84, 0x8d, 0x03, 0x37, 0x09, 0x81, 0x5c, 0x14, 0x80, 0xb8, 0x08, 0x80,
      0xcb, 0x2a, 0x38, 0x03, 0x0a, 0x06, 0x38, 0x08, 0x46, 0x08, 0x0c, 0x06,
      0x74, 0x0b, 0x1e, 0x03, 0x5a, 0x04, 0x59, 0x09, 0x80, 0x83, 0x18, 0x1c,
      0x0a, 0x16, 0x09, 0x4c, 0x04, 0x80, 0x8a, 0x06, 0xab, 0xa4, 0x0c, 0x17,
      0x04, 0x31, 0xa1, 0x04, 0x81, 0xda, 0x26, 0x07, 0x0c, 0x05, 0x05, 0x80,
      0xa5, 0x11, 0x81, 0x6d, 0x10, 0x78, 0x28, 0x2a, 0x06, 0x4c, 0x04, 0x80,
      0x8d, 0x04, 0x80, 0xbe, 0x03, 0x1b, 0x03, 0x0f, 0x0d,
  };
  static constexpr unsigned char normal1[] = {
      0x5e, 0x22, 0x7b, 0x05, 0x03, 0x04, 0x2d, 0x03, 0x66, 0x03, 0x01, 0x2f,
      0x2e, 0x80, 0x82, 0x1d, 0x03, 0x31, 0x0f, 0x1c, 0x04, 0x24, 0x09, 0x1e,
      0x05, 0x2b, 0x05, 0x44, 0x04, 0x0e, 0x2a, 0x80, 0xaa, 0x06, 0x24, 0x04,
      0x24, 0x04, 0x28, 0x08, 0x34, 0x0b, 0x01, 0x80, 0x90, 0x81, 0x37, 0x09,
      0x16, 0x0a, 0x08, 0x80, 0x98, 0x39, 0x03, 0x63, 0x08, 0x09, 0x30, 0x16,
      0x05, 0x21, 0x03, 0x1b, 0x05, 0x01, 0x40, 0x38, 0x04, 0x4b, 0x05, 0x2f,
      0x04, 0x0a, 0x07, 0x09, 0x07, 0x40, 0x20, 0x27, 0x04, 0x0c, 0x09, 0x36,
      0x03, 0x3a, 0x05, 0x1a, 0x07, 0x04, 0x0c, 0x07, 0x50, 0x49, 0x37, 0x33,
      0x0d, 0x33, 0x07, 0x2e, 0x08, 0x0a, 0x81, 0x26, 0x52, 0x4e, 0x28, 0x08,
      0x2a, 0x56, 0x1c, 0x14, 0x17, 0x09, 0x4e, 0x04, 0x1e, 0x0f, 0x43, 0x0e,
      0x19, 0x07, 0x0a, 0x06, 0x48, 0x08, 0x27, 0x09, 0x75, 0x0b, 0x3f, 0x41,
      0x2a, 0x06, 0x3b, 0x05, 0x0a, 0x06, 0x51, 0x06, 0x01, 0x05, 0x10, 0x03,
      0x05, 0x80, 0x8b, 0x62, 0x1e, 0x48, 0x08, 0x0a, 0x80, 0xa6, 0x5e, 0x22,
      0x45, 0x0b, 0x0a, 0x06, 0x0d, 0x13, 0x39, 0x07, 0x0a, 0x36, 0x2c, 0x04,
      0x10, 0x80, 0xc0, 0x3c, 0x64, 0x53, 0x0c, 0x48, 0x09, 0x0a, 0x46, 0x45,
      0x1b, 0x48, 0x08, 0x53, 0x1d, 0x39, 0x81, 0x07, 0x46, 0x0a, 0x1d, 0x03,
      0x47, 0x49, 0x37, 0x03, 0x0e, 0x08, 0x0a, 0x06, 0x39, 0x07, 0x0a, 0x81,
      0x36, 0x19, 0x80, 0xb7, 0x01, 0x0f, 0x32, 0x0d, 0x83, 0x9b, 0x66, 0x75,
      0x0b, 0x80, 0xc4, 0x8a, 0xbc, 0x84, 0x2f, 0x8f, 0xd1, 0x82, 0x47, 0xa1,
      0xb9, 0x82, 0x39, 0x07, 0x2a, 0x04, 0x02, 0x60, 0x26, 0x0a, 0x46, 0x0a,
      0x28, 0x05, 0x13, 0x82, 0xb0, 0x5b, 0x65, 0x4b, 0x04, 0x39, 0x07, 0x11,
      0x40, 0x05, 0x0b, 0x02, 0x0e, 0x97, 0xf8, 0x08, 0x84, 0xd6, 0x2a, 0x09,
      0xa2, 0xf7, 0x81, 0x1f, 0x31, 0x03, 0x11, 0x04, 0x08, 0x81, 0x8c, 0x89,
      0x04, 0x6b, 0x05, 0x0d, 0x03, 0x09, 0x07, 0x10, 0x93, 0x60, 0x80, 0xf6,
      0x0a, 0x73, 0x08, 0x6e, 0x17, 0x46, 0x80, 0x9a, 0x14, 0x0c, 0x57, 0x09,
      0x19, 0x80, 0x87, 0x81, 0x47, 0x03, 0x85, 0x42, 0x0f, 0x15, 0x85, 0x50,
      0x2b, 0x80, 0xd5, 0x2d, 0x03, 0x1a, 0x04, 0x02, 0x81, 0x70, 0x3a, 0x05,
      0x01, 0x85, 0x00, 0x80, 0xd7, 0x29, 0x4c, 0x04, 0x0a, 0x04, 0x02, 0x83,
      0x11, 0x44, 0x4c, 0x3d, 0x80, 0xc2, 0x3c, 0x06, 0x01, 0x04, 0x55, 0x05,
      0x1b, 0x34, 0x02, 0x81, 0x0e, 0x2c, 0x04, 0x64, 0x0c, 0x56, 0x0a, 0x80,
      0xae, 0x38, 0x1d, 0x0d, 0x2c, 0x04, 0x09, 0x07, 0x02, 0x0e, 0x06, 0x80,
      0x9a, 0x83, 0xd8, 0x08, 0x0d, 0x03, 0x0d, 0x03, 0x74, 0x0c, 0x59, 0x07,
      0x0c, 0x14, 0x0c, 0x04, 0x38, 0x08, 0x0a, 0x06, 0x28, 0x08, 0x22, 0x4e,
      0x81, 0x54, 0x0c, 0x15, 0x03, 0x03, 0x05, 0x07, 0x09, 0x19, 0x07, 0x07,
      0x09, 0x03, 0x0d, 0x07, 0x29, 0x80, 0xcb, 0x25, 0x0a, 0x84, 0x06,
  };
  auto lower = static_cast<uint16_t>(cp);
  if (cp < 0x10000) {
    return is_printable(lower, singletons0,
                        sizeof(singletons0) / sizeof(*singletons0),
                        singletons0_lower, normal0, sizeof(normal0));
  }
  if (cp < 0x20000) {
    return is_printable(lower, singletons1,
                        sizeof(singletons1) / sizeof(*singletons1),
                        singletons1_lower, normal1, sizeof(normal1));
  }
  if (0x2a6de <= cp && cp < 0x2a700) return false;
  if (0x2b735 <= cp && cp < 0x2b740) return false;
  if (0x2b81e <= cp && cp < 0x2b820) return false;
  if (0x2cea2 <= cp && cp < 0x2ceb0) return false;
  if (0x2ebe1 <= cp && cp < 0x2f800) return false;
  if (0x2fa1e <= cp && cp < 0x30000) return false;
  if (0x3134b <= cp && cp < 0xe0100) return false;
  if (0xe01f0 <= cp && cp < 0x110000) return false;
  return cp < 0x110000;
}

}  // namespace detail

FMT_END_NAMESPACE

#endif  // FMT_FORMAT_INL_H_
