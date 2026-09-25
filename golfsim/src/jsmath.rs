//! JavaScript's numerics, bit for bit as Node (V8 12.x) computes them, so the
//! port reproduces the game exactly rather than approximately.
//!
//! - `hypot2`/`hypot3`: V8's own algorithm (normalise by the max, Kahan sum).
//! - `exp`, `atan2`: fdlibm, which the `libm` crate reproduces exactly.
//! - `log`, `log1p`, `expm1`, `tanh`: fdlibm as V8 ships it (src/base/ieee754.cc),
//!   ported below. The `libm` crate's versions are newer musl code and differ.
//! - `sin`, `cos`: fdlibm too (Node's V8 is built without its optional glibc
//!   trig). Not correctly rounded, so only the same algorithm gives the same
//!   bits: the `libm` crate reduces arguments differently and misses ~1%.
//!   `rl/test/parity.test.mjs` checks all of this against Node.
//! - `RandomGenerator` (LittleJS's seeded xorshift) and `mulberry32` (the RL
//!   env's Math.random) are integer-exact.

// A verbatim port: fdlibm constants as printed, `x - x` for NaN, JS clamp semantics
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::eq_op,
    clippy::manual_clamp,
    clippy::needless_late_init,
    clippy::should_implement_trait
)]

/// ToInt32, as every JS bitwise operator applies it.
pub fn to_int32(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    let t = x.trunc();
    let m = t.rem_euclid(4294967296.0);
    m as u64 as u32 as i32
}

/// `clamp(v, 0, 1)` from engineMath.js.
#[inline]
pub fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// engineMath's lerp: the percent is clamped.
#[inline]
pub fn lerp(a: f64, b: f64, p: f64) -> f64 {
    a + clamp01(p) * (b - a)
}

#[inline]
pub fn smooth_step(p: f64) -> f64 {
    let p = clamp01(p);
    p * p * (3.0 - 2.0 * p)
}

/// Math.min(a, b) for non-NaN inputs, including the sign of zero.
#[inline]
pub fn js_min(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else if b < a {
        b
    } else if a == 0.0 && (a.is_sign_negative() || b.is_sign_negative()) {
        -0.0
    } else {
        a
    }
}

/// Math.max(a, b) for non-NaN inputs, including the sign of zero.
#[inline]
pub fn js_max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else if b > a {
        b
    } else if a == 0.0 && (a.is_sign_positive() || b.is_sign_positive()) {
        0.0
    } else {
        a
    }
}

/// Math.hypot(a, b), V8's algorithm.
#[inline]
pub fn hypot2(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    let m = if b > a { b } else { a };
    if m == 0.0 {
        return 0.0;
    }
    if m == f64::INFINITY {
        return m;
    }
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for v in [a, b] {
        let n = v / m;
        let su = n * n - c;
        let p = s + su;
        c = (p - s) - su;
        s = p;
    }
    s.sqrt() * m
}

/// Math.hypot(a, b, c), V8's algorithm.
#[inline]
pub fn hypot3(a: f64, b: f64, c: f64) -> f64 {
    let (a, b, c) = (a.abs(), b.abs(), c.abs());
    let mut m = 0.0;
    for v in [a, b, c] {
        if v > m {
            m = v;
        }
    }
    if m == 0.0 {
        return 0.0;
    }
    if m == f64::INFINITY {
        return m;
    }
    let (mut s, mut k) = (0.0f64, 0.0f64);
    for v in [a, b, c] {
        let n = v / m;
        let su = n * n - k;
        let p = s + su;
        k = (p - s) - su;
        s = p;
    }
    s.sqrt() * m
}

#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

///////////////////////////////////////////////////////////////////////////////
// fdlibm (Sun Microsystems, 1993: "Permission to use, copy, modify, and
// distribute this software is freely granted, provided that this notice is
// preserved."), as in V8's src/base/ieee754.cc.

#[inline]
fn high(x: f64) -> i32 {
    (x.to_bits() >> 32) as u32 as i32
}

#[inline]
fn low(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline]
fn with_high(x: f64, hi: i32) -> f64 {
    f64::from_bits(((hi as u32 as u64) << 32) | (x.to_bits() & 0xFFFF_FFFF))
}

#[inline]
fn from_words(hi: i32, lo: u32) -> f64 {
    f64::from_bits(((hi as u32 as u64) << 32) | lo as u64)
}

const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const TWO54: f64 = 1.80143985094819840000e+16;
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

pub fn log(mut x: f64) -> f64 {
    let mut hx = high(x);
    let lx = low(x);
    let mut k: i32 = 0;
    if hx < 0x00100000 {
        if ((hx & 0x7FFFFFFF) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = high(x);
    }
    if hx >= 0x7FF00000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000FFFFF;
    let i = (hx + 0x95F64) & 0x100000;
    x = with_high(x, hx | (i ^ 0x3FF00000));
    k += i >> 20;
    let f = x - 1.0;
    if (0x000FFFFF & (2 + hx)) < 3 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f64;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.33333333333333333 * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let mut i = hx - 0x6147A;
    let w = z * z;
    let j = 0x6B851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

pub fn log1p(x: f64) -> f64 {
    let hx = high(x);
    let ax = hx & 0x7FFFFFFF;
    let mut k: i32 = 1;
    let mut f = 0.0;
    let mut hu: i32 = 0;
    let mut c = 0.0;
    if hx < 0x3FDA827A {
        if ax >= 0x3FF00000 {
            return if x == -1.0 { f64::NEG_INFINITY } else { f64::NAN };
        }
        if ax < 0x3E200000 {
            if TWO54 + x > 0.0 && ax < 0x3C900000 {
                return x;
            }
            return x - x * x * 0.5;
        }
        if hx > 0 || hx <= 0xBFD2BEC4u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    }
    if hx >= 0x7FF00000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x43400000 {
            u = 1.0 + x;
            hu = high(u);
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = high(u);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000FFFFF;
        if hu < 0x6A09E {
            u = with_high(u, hu | 0x3FF00000);
        } else {
            k += 1;
            u = with_high(u, hu | 0x3FE00000);
            hu = (0x00100000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let kd = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c += kd * LN2_LO;
            return kd * LN2_HI + c;
        }
        let r = hfsq * (1.0 - 0.66666666666666666 * f);
        return if k == 0 {
            f - r
        } else {
            kd * LN2_HI - ((r - (kd * LN2_LO + c)) - f)
        };
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let r = z * (LG1 + z * (LG2 + z * (LG3 + z * (LG4 + z * (LG5 + z * (LG6 + z * LG7))))));
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        kd * LN2_HI - ((hfsq - (s * (hfsq + r) + (kd * LN2_LO + c))) - f)
    }
}

pub fn expm1(mut x: f64) -> f64 {
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const Q1: f64 = -3.33333333333331316428e-02;
    const Q2: f64 = 1.58730158725481460165e-03;
    const Q3: f64 = -7.93650757867487942473e-05;
    const Q4: f64 = 4.00821782732936239552e-06;
    const Q5: f64 = -2.01099218183624371326e-07;
    let mut hx = high(x) as u32;
    let xsb = hx & 0x80000000;
    hx &= 0x7FFFFFFF;
    if hx >= 0x4043687A {
        if hx >= 0x40862E42 {
            if hx >= 0x7FF00000 {
                if ((hx & 0xFFFFF) | low(x)) != 0 {
                    return x + x;
                }
                return if xsb == 0 { x } else { -1.0 };
            }
            if x > O_THRESHOLD {
                return f64::INFINITY;
            }
        }
        if xsb != 0 {
            return -1.0;
        }
    }
    let k: i32;
    let mut c = 0.0;
    if hx > 0x3FD62E42 {
        let (hi, lo);
        if hx < 0x3FF0A2B2 {
            if xsb == 0 {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi = x - t * LN2_HI;
            lo = t * LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3C900000 {
        return x;
    } else {
        k = 0;
    }
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs);
    }
    let twopk = from_words(0x3FF00000 + ((k as u32) << 20) as i32, 0);
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        return if x < -0.25 {
            -2.0 * (e - (x + 0.5))
        } else {
            1.0 + 2.0 * (x - e)
        };
    }
    if k <= -2 || k > 56 {
        let mut y = 1.0 - (e - x);
        if k == 1024 {
            y = y * 2.0 * 8.98846567431158e+307;
        } else {
            y *= twopk;
        }
        return y - 1.0;
    }
    let y;
    if k < 20 {
        let t = from_words(0x3FF00000 - (0x200000 >> k), 0);
        y = (t - (e - x)) * twopk;
    } else {
        let t = from_words((0x3FF - k) << 20, 0);
        y = ((x - (e + t)) + 1.0) * twopk;
    }
    y
}

pub fn tanh(x: f64) -> f64 {
    let jx = high(x);
    let ix = jx & 0x7FFFFFFF;
    if ix >= 0x7FF00000 {
        return if jx >= 0 { 1.0 / x + 1.0 } else { 1.0 / x - 1.0 };
    }
    let z;
    if ix < 0x40360000 {
        if ix < 0x3E300000 {
            return x;
        }
        if ix >= 0x3FF00000 {
            let t = expm1(2.0 * x.abs());
            z = 1.0 - 2.0 / (t + 2.0);
        } else {
            let t = expm1(-2.0 * x.abs());
            z = -t / (t + 2.0);
        }
    } else {
        z = 1.0;
    }
    if jx >= 0 {
        z
    } else {
        -z
    }
}

/// __ieee754_rem_pio2 for |x| <= 2^19 * pi/2 (the "medium size" path; the
/// game's arguments stay far below it). Returns (n, y0, y1).
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    const INVPIO2: f64 = 6.36619772367581382433e-01;
    const PIO2_1: f64 = 1.57079632673412561417e+00;
    const PIO2_1T: f64 = 6.07710050650619224932e-11;
    const PIO2_2: f64 = 6.07710050630396597660e-11;
    const PIO2_2T: f64 = 2.02226624879595063154e-21;
    const PIO2_3: f64 = 2.02226624871116645580e-21;
    const PIO2_3T: f64 = 8.47842766036889956997e-32;
    const NPIO2_HW: [i32; 32] = [
        0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A,
        0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
        0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC,
        0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
    ];
    let hx = high(x);
    let ix = hx & 0x7FFFFFFF;
    if ix < 0x4002D97C {
        // |x| < 3pi/4, n = +-1
        let (y0, y1);
        if hx > 0 {
            let mut z = x - PIO2_1;
            if ix != 0x3FF921FB {
                y0 = z - PIO2_1T;
                y1 = (z - y0) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y0 = z - PIO2_2T;
                y1 = (z - y0) - PIO2_2T;
            }
            return (1, y0, y1);
        }
        let mut z = x + PIO2_1;
        if ix != 0x3FF921FB {
            y0 = z + PIO2_1T;
            y1 = (z - y0) + PIO2_1T;
        } else {
            z += PIO2_2;
            y0 = z + PIO2_2T;
            y1 = (z - y0) + PIO2_2T;
        }
        return (-1, y0, y1);
    }
    let t = x.abs();
    let n = (t * INVPIO2 + 0.5) as i32;
    let fnn = n as f64;
    let mut r = t - fnn * PIO2_1;
    let mut w = fnn * PIO2_1T;
    let mut y0;
    if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
        y0 = r - w;
    } else {
        let j = ix >> 20;
        y0 = r - w;
        let mut i = j - ((high(y0) >> 20) & 0x7FF);
        if i > 16 {
            let t = r;
            w = fnn * PIO2_2;
            r = t - w;
            w = fnn * PIO2_2T - ((t - r) - w);
            y0 = r - w;
            i = j - ((high(y0) >> 20) & 0x7FF);
            if i > 49 {
                let t = r;
                w = fnn * PIO2_3;
                r = t - w;
                w = fnn * PIO2_3T - ((t - r) - w);
                y0 = r - w;
            }
        }
    }
    let y1 = (r - y0) - w;
    if hx < 0 {
        (-n, -y0, -y1)
    } else {
        (n, y0, y1)
    }
}

#[inline]
fn kernel_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;
    let ix = high(x) & 0x7FFFFFFF;
    if ix < 0x3E400000 && x as i32 == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD33333 {
        return 1.0 - (0.5 * z - (z * r - x * y));
    }
    let qx = if ix > 0x3FE90000 {
        0.28125
    } else {
        from_words(ix - 0x00200000, 0)
    };
    let iz = 0.5 * z - qx;
    let a = 1.0 - qx;
    a - (iz - (z * r - x * y))
}

#[inline]
fn kernel_sin(x: f64, y: f64, iy: bool) -> f64 {
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;
    let ix = high(x) & 0x7FFFFFFF;
    if ix < 0x3E400000 && x as i32 == 0 {
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if !iy {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

const RP_MAX: i32 = 0x413921FB; // 2^19 * pi/2

pub fn sin(x: f64) -> f64 {
    let ix = high(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return kernel_sin(x, 0.0, false);
    }
    if ix >= 0x7FF00000 {
        return x - x;
    }
    if ix > RP_MAX {
        return libm::sin(x);
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => kernel_sin(y0, y1, true),
        1 => kernel_cos(y0, y1),
        2 => -kernel_sin(y0, y1, true),
        _ => -kernel_cos(y0, y1),
    }
}

pub fn cos(x: f64) -> f64 {
    let ix = high(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7FF00000 {
        return x - x;
    }
    if ix > RP_MAX {
        return libm::cos(x);
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => kernel_cos(y0, y1),
        1 => -kernel_sin(y0, y1, true),
        2 => -kernel_cos(y0, y1),
        _ => kernel_sin(y0, y1, true),
    }
}

///////////////////////////////////////////////////////////////////////////////
// RNGs

/// LittleJS's RandomGenerator (engineMath.js): xorshift over ToInt32.
#[derive(Clone)]
pub struct RandomGenerator {
    seed: i32,
}

impl RandomGenerator {
    pub fn new(seed: f64) -> Self {
        RandomGenerator { seed: to_int32(seed) }
    }

    /// `float(a = 1, b = 0)`: b + (a - b) * u
    #[inline]
    pub fn float(&mut self, a: f64, b: f64) -> f64 {
        let mut s = self.seed;
        s ^= s.wrapping_shl(13);
        s ^= ((s as u32) >> 17) as i32;
        s ^= s.wrapping_shl(5);
        self.seed = s;
        b + (a - b) * ((s as u32) as f64 / 4294967296.0)
    }
    #[inline]
    pub fn unit(&mut self) -> f64 {
        self.float(1.0, 0.0)
    }
    #[inline]
    pub fn int(&mut self, a: f64) -> f64 {
        self.float(a, 0.0).floor()
    }
    #[inline]
    pub fn bool(&mut self, chance: f64) -> bool {
        self.unit() < chance
    }
    #[inline]
    pub fn sign(&mut self) -> f64 {
        if self.unit() > 0.5 {
            1.0
        } else {
            -1.0
        }
    }
    #[inline]
    pub fn float_sign(&mut self, a: f64, b: f64) -> f64 {
        let f = self.float(a, b);
        f * self.sign()
    }
}

/// mulberry32, rl/sim/loader.mjs: the seeded Math.random of an env instance.
#[derive(Clone)]
pub struct Mulberry32 {
    a: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Mulberry32 { a: seed }
    }
    #[inline]
    pub fn next(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6D2B79F5);
        let mut t = self.a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int32() {
        assert_eq!(to_int32(4294967296.0 + 5.0), 5);
        assert_eq!(to_int32(2147483648.0), i32::MIN);
        assert_eq!(to_int32(-1.5), -1);
    }

    #[test]
    fn sin_cos_basics() {
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        for i in 0..1000 {
            let x = (i as f64 - 500.0) * 0.37;
            assert!((sin(x) - x.sin()).abs() < 1e-15);
            assert!((cos(x) - x.cos()).abs() < 1e-15);
        }
    }

    #[test]
    fn fdlibm_close_to_std() {
        for i in 1..2000 {
            let x = i as f64 * 0.173;
            assert!((log(x) - x.ln()).abs() < 1e-14 * x.ln().abs().max(1.0));
            assert!((log1p(x) - x.ln_1p()).abs() < 1e-14 * x.ln_1p().abs().max(1.0));
            let y = (i as f64 - 1000.0) / 100.0;
            assert!((tanh(y) - y.tanh()).abs() < 1e-15);
        }
    }

    #[test]
    fn mulberry_first_values() {
        // node -e "…mulberry32(1)…" : 0.6270739405881613, 0.002735721180215478
        let mut m = Mulberry32::new(1);
        assert_eq!(m.next(), 0.6270739405881613);
        assert_eq!(m.next(), 0.002735721180215478);
    }
}
