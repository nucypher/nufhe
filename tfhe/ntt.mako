<%def name="ffp_shared()">

enum ConvKind: int32_t {
  LINEAR_CONVOLUTION =  0,
  POSITIVE_CYCLIC_CONVOLUTION =  1,
  NEGATIVE_CYCLIC_CONVOLUTION = -1
};

__host__ __device__ inline
constexpr uint32_t Log2Const(uint32_t x) {
  return x < 2 ? 0 : 1 + Log2Const(x / 2);
}


__device__ inline
uint32_t ThisThreadRankInBlock() {
  return virtual_local_id(1);
  ## threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

template <uint32_t dim_x, uint32_t dim_y, uint32_t dim_z>
__device__ inline
void Index3DFrom1D(uint3& t3d, uint32_t t1d) {
  t3d.x = t1d % dim_x;
  t1d /= dim_x;
  t3d.y = t1d % dim_y;
  t3d.z = t1d / dim_y;
}

class FFP {

private:

  /** Field modulus P. */
  static const uint64_t kModulus_ = 0xffffffff00000001UL;

  /** A 2^32-th primitive root of unity mod P. */
  static const uint64_t kRoot2e32_ = 0xa70dc47e4cbdf43fUL;

  /** An 64-bit unsigned integer within [0, P-1]. */
  uint64_t val_;

  /** A 64-bit integer modulo P. */
  /** Cannot avoid divergence, since comparison causes branches in CUDA. */
  __device__ inline
  void ModP(uint64_t& x) {
    asm("{\n\t"
        ".reg .u32        m;\n\t"
        ".reg .u64        t;\n\t"
        "set.ge.u32.u64   m, %0, %1;\n\t"
        "mov.b64          t, {m, 0};\n\t"
        "add.u64         %0, %0, t;\n\t"
        "}"
        : "+l"(x)
        : "l"(kModulus_));
  }

public:

  //////////////////////////////////////////////////////////////////////////////
  // Access and set value

  /** Default constructor. Value is not specified. */
  __host__ __device__ inline
  FFP() {} // shared memory requires empty constructor

  /**
   * Setting value to a (mod P).
   * @param[in] a An unsigned integer in [0, P-1].
   *            Immediates are uint32_t unless specified.
   */
  __host__ __device__ inline
  FFP(uint8_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(uint16_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(uint32_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(uint64_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(int8_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  __host__ __device__ inline
  FFP(int16_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  __host__ __device__ inline
  FFP(int32_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  __host__ __device__ inline
  FFP(int64_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  /** Default destructor. Value is not wiped. */
  __host__ __device__ inline
  ~FFP() {}

  /** Get value. */
  __host__ __device__ inline
  uint64_t& val() { return val_; }

  /** Get value. */
  __host__ __device__ inline
  const uint64_t& val() const { return val_; }

  /** Return modulus P. */
  __host__ __device__ inline
  static uint64_t kModulus() { return kModulus_; };

  /** Return 2^32-th primitive root of unity mod P. */
  __host__ __device__ inline
  static uint64_t kRoot2e32() { return kRoot2e32_; };

  //////////////////////////////////////////////////////////////////////////////
  // Operators

  /**
   * Assign.
   * @param a [description]
   */
  __host__ __device__ inline
  FFP& operator=(uint8_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(uint16_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(uint32_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(uint64_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(int8_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(int16_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(int32_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(int64_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(FFP a) { this->val_ = a.val(); return *this; }

  /** Explicit conversion. */
  __host__ __device__ inline
  explicit operator uint64_t() { return val_; } // correct result

  __host__ __device__ inline
  explicit operator uint8_t() { return (uint8_t)val_; } // truncated result

  __host__ __device__ inline
  explicit operator uint16_t() { return (uint16_t)val_; } // truncated result

  __host__ __device__ inline
  explicit operator uint32_t() { return (uint32_t)val_; } // truncated result

  // !!! Added for simplicity
  // !!! Not sure what it does if val_ is not actually inside the i32 range.
  __host__ __device__ inline
  explicit operator int32_t() {
    uint64_t med = FFP::kModulus() / 2;
    return (int32_t)val_ - (val_ > med);
  }

  /** Addition in FF(P): val_ = val_ + a mod P. */
  __device__ inline
  FFP& operator+=(const FFP& a) { this->Add(*this, a); return *this; }

  /** Addition in FF(P): return a + b mod P. */
  friend __device__ inline
  FFP operator+(const FFP& a, const FFP& b) { FFP r; r.Add(a, b); return r; }

  /** Subtraction in FF(P): val_ = val_ - a mod P. */
  __device__ inline
  FFP& operator-=(const FFP& a) { this->Sub(*this, a); return *this; }

  /** Subtraction in FF(P): return a - b mod P. */
  friend __device__ inline
  FFP operator-(const FFP& a, const FFP& b) { FFP r; r.Sub(a, b); return r; }

  /** Multiplication in FF(P): val_ = val_ * a mod P. */
  __device__ inline
  FFP& operator*=(const FFP& a) { this->Mul(*this, a); return *this; }

  /** Multiplication in FF(P): return a * b mod P. */
  friend __device__ inline
  FFP operator*(const FFP& a, const FFP& b) { FFP r; r.Mul(a, b); return r; }

  /** Equality. */
  __host__ __device__ inline
  bool operator==(const FFP& other) { return (bool)(val_ == other.val()); }

  /** Inequality. */
  __host__ __device__ inline
  bool operator!=(const FFP& other) { return (bool)(val_ != other.val()); }

  //////////////////////////////////////////////////////////////////////////////
  // Miscellaneous

  /**
   * Return a primitive n-th root in FF(P): val_ ^ n = 1 mod P.
   * @param[in] n A power of 2.
   */
  __device__ inline
  static FFP Root(uint32_t n) {
    return Pow(kRoot2e32_, (uint32_t)((0x1UL << 32) / n));
  }

  /**
   * Return the inverse of 2^log_n in FF(P): 2^{-log_n} mod P.
   * @param log_n An integer in [0, 32]
   */
  __host__ __device__ inline
  static FFP InvPow2(uint32_t log_n) {
    uint32_t r[2];
    r[0] = (0x1 << (32 - log_n)) + 1;
    r[1] = -r[0];
    return FFP(*(uint64_t*)r);
  }

  /** Exchange values with a. */
  __host__ __device__ inline
  void Swap(FFP& a) {
    uint64_t t = val_;
    val_ = a.val_;
    a.val_ = t;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Arithmetic

  /** Addition in FF(P): val_ = a + b mod P. */
  __device__ inline
  void Add(const FFP& a, const FFP& b) {
    asm("{\n\t"
        ".reg .u32          m;\n\t"
        ".reg .u64          t;\n\t"
        ".reg .pred         p;\n\t"
        // this = a + b;
        "add.u64            %0, %1, %2;\n\t"
        // this += (uint32_t)(-(this < b || this >= FFP_MODULUS));
        "setp.lt.u64        p, %0, %2;\n\t"
        "set.ge.or.u32.u64  m, %0, %3, p;\n\t"
        "mov.b64            t, {m, 0};\n\t"
        "add.u64            %0, %0, t;\n\t"
        "}"
        : "+l"(val_)
        : "l"(a.val_), "l"(b.val_), "l"(kModulus_));
  }

  /** Subtraction in FF(P): val_ = a + b mod P. */
  __device__ inline
  void Sub(const FFP& a, const FFP& b) {
    register uint64_t r = 0;
    asm("{\n\t"
        ".reg .u32          m;\n\t"
        ".reg .u64          t;\n\t"
        // this = a - b;
        "sub.u64            %0, %1, %2;\n\t"
        // this -= (uint32_t)(-(this > a));
        "set.gt.u32.u64     m, %0, %1;\n\t"
        "mov.b64            t, {m, 0};\n\t"
        "sub.u64            %0, %0, t;\n\t"
        "}"
        : "+l"(r)
        : "l"(a.val_), "l"(b.val_));
    val_ = r;
  }

  /** Multiplication in FF(P): val_ = a * b mod P. */
  __device__ inline
  void Mul(const FFP& a, const FFP& b) {
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          m0, m1, m2, m3;\n\t"
        ".reg .u64          t;\n\t"
        ".reg .pred         p, q;\n\t"
        // 128-bit = 64-bit * 64-bit
        "mul.lo.u64         t, %1, %2;\n\t"
        "mov.b64            {m0, m1}, t;\n\t"
        "mul.hi.u64         t, %1, %2;\n\t"
        "mov.b64            {m2, m3}, t;\n\t"
        // 128-bit mod P with add / sub
        "add.u32            r1, m1, m2;\n\t"
        "sub.cc.u32         r0, m0, m2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, m3;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // fix result
        "setp.eq.u32        p|q, m2, 0;\n\t"
        "mov.b64            t, {m0, m1};\n\t"
        // ret -= (uint32_t)(-(ret > mul[0] && m[2] == 0));
        "set.gt.and.u32.u64 m3, %0, t, p;\n\t"
        "sub.cc.u32         r0, r0, m3;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < mul[0] && m[2] != 0));
        "set.lt.and.u32.u64 m3, %0, t, q;\n\t"
        "add.cc.u32         r0, r0, m3;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "l"(a.val_), "l"(b.val_));
    ModP(val_);
  }

  /** \brief Exponentiation in FF(P): val_ = val_ ^ e mod P. */
  __device__ inline
  void Pow(uint32_t e) {
    if (0 == e) {
      val_ = 1;
      return;
    }
    FFP y = 1;
    uint64_t n = (uint64_t)e;
    while (n > 1) {
      if (0 != (n & 0x1))
        y *= (*this);
      *this *= (*this);
      n >>= 1;
    }
    *this *= y;
  }

  /** \brief Exponentiation in FF(P): return a ^ e mod P. */
  __device__ inline
  static FFP Pow(const FFP& a, uint32_t e) {
    FFP r = a;
    r.Pow(e);
    return r;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [0, 32)
   */
  __device__ inline
  void Lsh32(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x >> (64-l));
        // t[1] = (uint32_t)(x >> (32-l));
        // t[0] = (uint32_t)(x << l);
        "mov.b64        {r0, r1}, %0;\n\t"
        "shl.b32        t0, r0, %1;\n\t"
        "sub.u32        n, 32, %1;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t1, t2}, s;\n\t"
        // mod P
        "add.u32        r1, t1, t2;\n\t"
        "sub.cc.u32     r0, t0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
        "mov.b64        s, {t0, t1};\n\t"
        "set.lt.u32.u64 t2, %0, s;\n\t"
        "add.cc.u32     r0, r0, t2;\n\t"
        "addc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [32, 64)
   */
  __device__ inline
  void Lsh64(uint32_t l) {
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          t0, t1, t2;\n\t"
        ".reg .u32          n;\n\t"
        ".reg .u64          s;\n\t"
        ".reg .pred         p, q;\n\t"
        // t[2] = (uint32_t)(x >> (96-l));
        // t[1] = (uint32_t)(x >> (64-l));
        // t[0] = (uint32_t)(x << (l-32));
        "mov.b64            {r0, r1}, %0;\n\t"
        "sub.u32            n, %1, 32;\n\t"
        "shl.b32            t0, r0, n;\n\t"
        "sub.u32            n, 32, n;\n\t"
        "shr.b64            s, %0, n;\n\t"
        "mov.b64            {t1, t2}, s;\n\t"
        // mod P
        "add.u32            r1, t0, t1;\n\t"
        "sub.cc.u32         r0, 0, t1;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
        "setp.eq.u32        p|q, t1, 0;\n\t"
        "mov.b64            s, {0, t0};\n\t"
        "set.gt.and.u32.u64 t2, %0, s, p;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
        "set.lt.and.u32.u64 t2, %0, s, q;\n\t"
        "add.cc.u32         r0, r0, t2;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [64, 96)
   */
  __device__ inline
  void Lsh96(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x >> (128-l));
        // t[1] = (uint32_t)(x >> (96-l));
        // t[0] = (uint32_t)(x << (l-64));
        "mov.b64        {r0, r1}, %0;\n\t"
        "sub.u32        n, %1, 64;\n\t"
        "shl.b32        t0, r0, n;\n\t"
        "sub.u32        n, 32, n;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t1, t2}, s;\n\t"
        // mod P
        "add.cc.u32     r0, t1, t0;\n\t"
        "addc.u32       r1, t2, 0;\n\t"
        "sub.u32        r1, r1, t0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret -= (uint32_t)(-(ret > ((uint64_t *)t)[1]));
        "mov.b64        s, {t1, t2};\n\t"
        "set.gt.u32.u64 t2, %0, s;\n\t"
        "sub.cc.u32     r0, r0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
    val_ = kModulus_ - val_;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [96, 128)
   */
  __device__ inline
  void Lsh128(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x >> (160-l));
        // t[1] = (uint32_t)(x >> (128-l));
        // t[0] = (uint32_t)(x << (l-96));
        "mov.b64        {r0, r1}, %0;\n\t"
        "sub.u32        n, %1, 96;\n\t"
        "shl.b32        t0, r0, n;\n\t"
        "sub.u32        n, 32, n;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t1, t2}, s;\n\t"
        // mod P
        "add.u32        r1, t1, t2;\n\t"
        "sub.cc.u32     r0, t0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
        "mov.b64        s, {t0, t1};\n\t"
        "set.lt.u32.u64 t2, %0, s;\n\t"
        "add.cc.u32     r0, r0, t2;\n\t"
        "addc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
    val_ = kModulus_ - val_;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [128, 160)
   */
  __device__ inline
  void Lsh160(uint32_t l) {
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          t0, t1, t2;\n\t"
        ".reg .u32          n;\n\t"
        ".reg .u64          s;\n\t"
        ".reg .pred         p, q;\n\t"
        // t[2] = (uint32_t)(x >> (192-l));
        // t[1] = (uint32_t)(x >> (160-l));
        // t[0] = (uint32_t)(x << (l-128));
        "mov.b64            {r0, r1}, %0;\n\t"
        "sub.u32            n, %1, 128;\n\t"
        "shl.b32            t0, r0, n;\n\t"
        "sub.u32            n, 32, n;\n\t"
        "shr.b64            s, %0, n;\n\t"
        "mov.b64            {t1, t2}, s;\n\t"
        // mod P
        "add.u32            r1, t0, t1;\n\t"
        "sub.cc.u32         r0, 0, t1;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
        "setp.eq.u32        p|q, t1, 0;\n\t"
        "mov.b64            s, {0, t0};\n\t"
        "set.gt.and.u32.u64 t2, %0, s, p;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
        "set.lt.and.u32.u64 t2, %0, s, q;\n\t"
        "add.cc.u32         r0, r0, t2;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
    val_ = kModulus_ - val_;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [160, 192)
   */
  __device__ inline
  void Lsh192(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x << (l-160));
        // t[1] = (uint32_t)(x >> (224-l));
        // t[0] = (uint32_t)(x >> (192-l));
        "mov.b64        {r0, r1}, %0;\n\t"
        "sub.u32        n, %1, 160;\n\t"
        "shl.b32        t2, r0, n;\n\t"
        "sub.u32        n, 32, n;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t0, t1}, s;\n\t"
        // mod P
        "add.cc.u32     r0, t0, t2;\n\t"
        "addc.u32       r1, t1, 0;\n\t"
        "sub.u32        r1, r1, t2;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret > ((uint64_t *)t)[0]));
        "mov.b64        s, {t0, t1};\n\t"
        "set.gt.u32.u64 t2, %0, s;\n\t"
        "sub.cc.u32     r0, r0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
  }

}; // class FFP

</%def>


<%def name="ntt1024_twiddle(kernel_declaration, twd, twd_inv)">

${ffp_shared()}

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    uint32_t n = 1024;
    uint32_t idx;
    uint32_t cid;
    FFP w = FFP::Root(n);
    uint32_t e;

    VSIZE_T tid0 = virtual_local_id(0);
    VSIZE_T tid1 = virtual_local_id(1);
    VSIZE_T tid2 = virtual_local_id(2);

    cid = (tid0 << 6) + (tid1 << 3) + tid2;
    for (int i = 0; i < 8; i ++) {
        e = (tid0 * 8 + tid1 / 4 * 4 + (tid2 % 4)) * (i * 8 + (tid1 % 4) * 2 + tid2 / 4);
        idx = (i * n / 8) + cid;

        FFP twd = FFP::Pow(w, e);
        FFP twd_inv = FFP::Pow(w, (n - e) % n);

        ${twd.store_idx}(idx, twd.val());
        ${twd_inv.store_idx}(idx, twd_inv.val());
    }
}
</%def>


<%def name="ntt1024_twiddle_sqrt(kernel_declaration, twd_sqrt, twd_sqrt_inv)">

${ffp_shared()}

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    uint32_t n = 1024;
    uint32_t idx = virtual_global_id(0);
    FFP w = FFP::Root(2 * n);
    FFP n_inv = FFP::InvPow2(10);

    FFP twd_sqrt = FFP::Pow(w, idx);
    FFP twd_sqrt_inv = FFP::Pow(w, (2 * n - idx) % (2 * n)) * n_inv;

    ${twd_sqrt.store_idx}(idx, twd_sqrt.val());
    ${twd_sqrt_inv.store_idx}(idx, twd_sqrt_inv.val());
}
</%def>


<%def name="ntt_shared()">

${ffp_shared()}

////////////////////////////////////////////////////////////////////////////////
//// NTT conversions of sizes 2, 4, 8 on a single thread over registers     ////
////////////////////////////////////////////////////////////////////////////////

__device__ inline
void NTT2(FFP& r0, FFP& r1) {
  register FFP t = r0 - r1;
  r0 += r1;
  r1 = t;
}

__device__ inline
void NTT2(FFP* r) {
  NTT2(r[0], r[1]);
}

__device__ inline
void NTTInv2(FFP& r0, FFP& r1) {
  NTT2(r0, r1);
}

__device__ inline
void NTTInv2(FFP* r) {
  NTT2(r);
}

__device__ inline
void NTT4(FFP& r0, FFP& r1, FFP& r2, FFP& r3) {
  NTT2(r0, r2);
  NTT2(r1, r3);
  r3.Lsh64(48);
  NTT2(r0, r1);
  NTT2(r2, r3);
  r1.Swap(r2);
}

__device__ inline
void NTT4(FFP* r) {
  NTT4(r[0], r[1], r[2], r[3]);
}

__device__ inline
void NTTInv4(FFP& r0, FFP& r1, FFP& r2, FFP& r3) {
  NTTInv2(r0, r2);
  NTTInv2(r1, r3);
  r3.Lsh160(144);
  NTTInv2(r0, r1);
  NTTInv2(r2, r3);
  r1.Swap(r2);
}

__device__ inline
void NTTInv4(FFP* r) {
  NTTInv4(r[0], r[1], r[2], r[3]);
}

__device__ inline
void NTT8(FFP* r) {
  NTT2(r[0], r[4]);
  NTT2(r[1], r[5]);
  NTT2(r[2], r[6]);
  NTT2(r[3], r[7]);
  r[5].Lsh32(24);
  r[6].Lsh64(48);
  r[7].Lsh96(72);
  // instead of calling NTT4 ...
  NTT2(r[0], r[2]);
  NTT2(r[1], r[3]);
  r[3].Lsh64(48);
  NTT2(r[4], r[6]);
  NTT2(r[5], r[7]);
  r[7].Lsh64(48);
  NTT2(r);
  NTT2(&r[2]);
  NTT2(&r[4]);
  NTT2(&r[6]);
  // ... we save 2 swaps (otherwise 4) here
  r[1].Swap(r[4]);
  r[3].Swap(r[6]);
}

__device__ inline
void NTTInv8(FFP* r) {
  NTTInv2(r[0], r[4]);
  NTTInv2(r[1], r[5]);
  NTTInv2(r[2], r[6]);
  NTTInv2(r[3], r[7]);
  r[5].Lsh192(168);
  r[6].Lsh160(144);
  r[7].Lsh128(120);
  // instead of calling NTT4 ...
  NTTInv2(r[0], r[2]);
  NTTInv2(r[1], r[3]);
  r[3].Lsh160(144);
  NTTInv2(r[4], r[6]);
  NTTInv2(r[5], r[7]);
  r[7].Lsh160(144);
  NTTInv2(r);
  NTTInv2(&r[2]);
  NTTInv2(&r[4]);
  NTTInv2(&r[6]);
  // ... we save 2 swaps (otherwise 4) here
  r[1].Swap(r[4]);
  r[3].Swap(r[6]);
}


template <uint32_t col>
__device__ inline
void NTT8x2Lsh(FFP* s);

template <>
__device__ inline
void NTT8x2Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTT8x2Lsh<1>(FFP* s) {
  s[1].Lsh32(12);
  s[2].Lsh32(24);
  s[3].Lsh64(36);
  s[4].Lsh64(48);
  s[5].Lsh64(60);
  s[6].Lsh96(72);
  s[7].Lsh96(84);
}

__device__ inline
void NTT8x2Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTT8x2Lsh<1>(s);
}

/** s[i] << 6 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTT8x4Lsh(FFP* s);

template <>
__device__ inline
void NTT8x4Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTT8x4Lsh<1>(FFP* s) {
  s[1].Lsh32(6);
  s[2].Lsh32(12);
  s[3].Lsh32(18);
  s[4].Lsh32(24);
  s[5].Lsh32(30);
  s[6].Lsh64(36);
  s[7].Lsh64(42);
}

template <>
__device__ inline
void NTT8x4Lsh<2>(FFP* s) {
  s[1].Lsh32(12);
  s[2].Lsh32(24);
  s[3].Lsh64(36);
  s[4].Lsh64(48);
  s[5].Lsh64(60);
  s[6].Lsh96(72);
  s[7].Lsh96(84);
}

template <>
__device__ inline
void NTT8x4Lsh<3>(FFP* s) {
  s[1].Lsh32(18);
  s[2].Lsh64(36);
  s[3].Lsh64(54);
  s[4].Lsh96(72);
  s[5].Lsh96(90);
  s[6].Lsh128(108);
  s[7].Lsh128(126);
}

__device__ inline
void NTT8x4Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTT8x4Lsh<1>(s);
  else if (2 == col)
    NTT8x4Lsh<2>(s);
  else if (3 == col)
    NTT8x4Lsh<3>(s);
}

/** s[i] << 3 * i * col mod P. */

template <uint32_t col>
__device__ inline
void NTT8x8Lsh(FFP* s);

template <>
__device__ inline
void NTT8x8Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTT8x8Lsh<1>(FFP* s) {
  s[1].Lsh32(3);
  s[2].Lsh32(6);
  s[3].Lsh32(9);
  s[4].Lsh32(12);
  s[5].Lsh32(15);
  s[6].Lsh32(18);
  s[7].Lsh32(21);
}

template <>
__device__ inline
void NTT8x8Lsh<2>(FFP* s) {
  s[1].Lsh32(6);
  s[2].Lsh32(12);
  s[3].Lsh32(18);
  s[4].Lsh32(24);
  s[5].Lsh32(30);
  s[6].Lsh64(36);
  s[7].Lsh64(42);
}

template <>
__device__ inline
void NTT8x8Lsh<3>(FFP* s) {
  s[1].Lsh32(9);
  s[2].Lsh32(18);
  s[3].Lsh32(27);
  s[4].Lsh64(36);
  s[5].Lsh64(45);
  s[6].Lsh64(54);
  s[7].Lsh64(63);
}

template <>
__device__ inline
void NTT8x8Lsh<4>(FFP* s) {
  s[1].Lsh32(12);
  s[2].Lsh32(24);
  s[3].Lsh64(36);
  s[4].Lsh64(48);
  s[5].Lsh64(60);
  s[6].Lsh96(72);
  s[7].Lsh96(84);
}

template <>
__device__ inline
void NTT8x8Lsh<5>(FFP* s) {
  s[1].Lsh32(15);
  s[2].Lsh32(30);
  s[3].Lsh64(45);
  s[4].Lsh64(60);
  s[5].Lsh96(75);
  s[6].Lsh96(90);
  s[7].Lsh128(105);
}

template <>
__device__ inline
void NTT8x8Lsh<6>(FFP* s) {
  s[1].Lsh32(18);
  s[2].Lsh64(36);
  s[3].Lsh64(54);
  s[4].Lsh96(72);
  s[5].Lsh96(90);
  s[6].Lsh128(108);
  s[7].Lsh128(126);
}

template <>
__device__ inline
void NTT8x8Lsh<7>(FFP* s) {
  s[1].Lsh32(21);
  s[2].Lsh64(42);
  s[3].Lsh64(63);
  s[4].Lsh96(84);
  s[5].Lsh128(105);
  s[6].Lsh128(126);
  s[7].Lsh160(147);
}

__device__ inline
void NTT8x8Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTT8x8Lsh<1>(s);
  else if (2 == col)
    NTT8x8Lsh<2>(s);
  else if (3 == col)
    NTT8x8Lsh<3>(s);
  else if (4 == col)
    NTT8x8Lsh<4>(s);
  else if (5 == col)
    NTT8x8Lsh<5>(s);
  else if (6 == col)
    NTT8x8Lsh<6>(s);
  else if (7 == col)
    NTT8x8Lsh<7>(s);
}

/** s[i] << 192 - 12 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTTInv8x2Lsh(FFP* s);

template <>
__device__ inline
void NTTInv8x2Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTTInv8x2Lsh<1>(FFP* s) {
  s[1].Lsh192(180);
  s[2].Lsh192(168);
  s[3].Lsh160(156);
  s[4].Lsh160(144);
  s[5].Lsh160(132);
  s[6].Lsh128(120);
  s[7].Lsh128(108);
}

__device__ inline
void NTTInv8x2Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTTInv8x2Lsh<1>(s);
}

/** s[i] << 192 - 6 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTTInv8x4Lsh(FFP* s);

template <>
__device__ inline
void NTTInv8x4Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTTInv8x4Lsh<1>(FFP* s) {
  s[1].Lsh192(186);
  s[2].Lsh192(180);
  s[3].Lsh192(174);
  s[4].Lsh192(168);
  s[5].Lsh192(162);
  s[6].Lsh160(156);
  s[7].Lsh160(150);
}

template <>
__device__ inline
void NTTInv8x4Lsh<2>(FFP* s) {
  s[1].Lsh192(180);
  s[2].Lsh192(168);
  s[3].Lsh160(156);
  s[4].Lsh160(144);
  s[5].Lsh160(132);
  s[6].Lsh128(120);
  s[7].Lsh128(108);
}

template <>
__device__ inline
void NTTInv8x4Lsh<3>(FFP* s) {
  s[1].Lsh192(174);
  s[2].Lsh160(156);
  s[3].Lsh160(138);
  s[4].Lsh128(120);
  s[5].Lsh128(102);
  s[6].Lsh96(84);
  s[7].Lsh96(66);
}

__device__ inline
void NTTInv8x4Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTTInv8x4Lsh<1>(s);
  else if (2 == col)
    NTTInv8x4Lsh<2>(s);
  else if (3 == col)
    NTTInv8x4Lsh<3>(s);
}

/** s[i] << 192 - 6 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTTInv8x8Lsh(FFP* s);

template <>
__device__ inline
void NTTInv8x8Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTTInv8x8Lsh<1>(FFP* s) {
  s[1].Lsh192(189);
  s[2].Lsh192(186);
  s[3].Lsh192(183);
  s[4].Lsh192(180);
  s[5].Lsh192(177);
  s[6].Lsh192(174);
  s[7].Lsh192(171);
}

template <>
__device__ inline
void NTTInv8x8Lsh<2>(FFP* s) {
  s[1].Lsh192(186);
  s[2].Lsh192(180);
  s[3].Lsh192(174);
  s[4].Lsh192(168);
  s[5].Lsh192(162);
  s[6].Lsh160(156);
  s[7].Lsh160(150);
}

template <>
__device__ inline
void NTTInv8x8Lsh<3>(FFP* s) {
  s[1].Lsh192(183);
  s[2].Lsh192(174);
  s[3].Lsh192(165);
  s[4].Lsh160(156);
  s[5].Lsh160(147);
  s[6].Lsh160(138);
  s[7].Lsh160(129);
}

template <>
__device__ inline
void NTTInv8x8Lsh<4>(FFP* s) {
  s[1].Lsh192(180);
  s[2].Lsh192(168);
  s[3].Lsh160(156);
  s[4].Lsh160(144);
  s[5].Lsh160(132);
  s[6].Lsh128(120);
  s[7].Lsh128(108);
}

template <>
__device__ inline
void NTTInv8x8Lsh<5>(FFP* s) {
  s[1].Lsh192(177);
  s[2].Lsh192(162);
  s[3].Lsh160(147);
  s[4].Lsh160(132);
  s[5].Lsh128(117);
  s[6].Lsh128(102);
  s[7].Lsh96(87);
}

template <>
__device__ inline
void NTTInv8x8Lsh<6>(FFP* s) {
  s[1].Lsh192(174);
  s[2].Lsh160(156);
  s[3].Lsh160(138);
  s[4].Lsh128(120);
  s[5].Lsh128(102);
  s[6].Lsh96(84);
  s[7].Lsh96(66);
}

template <>
__device__ inline
void NTTInv8x8Lsh<7>(FFP* s) {
  s[1].Lsh192(171);
  s[2].Lsh160(150);
  s[3].Lsh160(129);
  s[4].Lsh128(108);
  s[5].Lsh96(87);
  s[6].Lsh96(66);
  s[7].Lsh64(45);
}

__device__ inline
void NTTInv8x8Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTTInv8x8Lsh<1>(s);
  else if (2 == col)
    NTTInv8x8Lsh<2>(s);
  else if (3 == col)
    NTTInv8x8Lsh<3>(s);
  else if (4 == col)
    NTTInv8x8Lsh<4>(s);
  else if (5 == col)
    NTTInv8x8Lsh<5>(s);
  else if (6 == col)
    NTTInv8x8Lsh<6>(s);
  else if (7 == col)
    NTTInv8x8Lsh<7>(s);
}


__device__ inline
void NTT1024Core(FFP* r,
                 FFP* s,
                 const FFP* twd,
                 const FFP* twd_sqrt,
                 const uint32_t& t1d,
                 const uint3& t3d) {
  FFP *ptr = nullptr;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] *= twd_sqrt[(i << 7) | t1d]; // mult twiddle sqrt
  NTT8(r);
  NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
  __syncthreads();

  ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 6];
  NTT2(r);
  NTT2(r + 2);
  NTT2(r + 4);
  NTT2(r + 6);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 6] = r[i];
  __syncthreads();

  ptr = &s[t1d];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 7] * twd[(i << 7) | t1d]; // mult twiddle
  NTT8(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 7] = r[i];
  __syncthreads();

  ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 2];
  NTT8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTT8(r);
}


__device__ inline
void NTTInv1024Core(FFP* r,
                    FFP* s,
                    const FFP* twd_inv,
                    const FFP* twd_sqrt_inv,
                    const uint32_t& t1d,
                    const uint3& t3d) {

  FFP *ptr = nullptr;
  NTTInv8(r);
  NTTInv8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
  __syncthreads();

  ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 6];
  NTT2(r);
  NTT2(r + 2);
  NTT2(r + 4);
  NTT2(r + 6);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 6] = r[i];
  __syncthreads();

  ptr = &s[t1d];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 7] * twd_inv[(i << 7) | t1d]; // mult twiddle
  NTTInv8(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 7] = r[i];
  __syncthreads();

  ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 2];
  NTTInv8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTTInv8(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] *= twd_sqrt_inv[(i << 7) | t1d]; // mult twiddle sqrt
}


template <typename T>
__device__
void NTT1024(FFP* out,
             T* in,
             FFP* temp_shared,
             FFP* twd,
             FFP* twd_sqrt,
             uint32_t leading_thread) {
  uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = FFP((T)in[(i << 7) | t1d]);
  NTT1024Core(r, temp_shared, twd, twd_sqrt, t1d, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] = r[i];
}


template <typename T>
__device__
void NTTInv1024(T* out,
                FFP* in,
                FFP* temp_shared,
                FFP* twd_inv,
                FFP* twd_sqrt_inv,
                uint32_t leading_thread) {
  uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = in[(i << 7) | t1d];
  NTTInv1024Core(r, temp_shared, twd_inv, twd_sqrt_inv, t1d, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] = (T)r[i].val(); // here there was a conversion to i32, which seems error-prone, since T may not be i32. Moved to a separate operator.
}

</%def>


<%def name="ntt1024(kernel_declaration, output, input_, twd, twd_sqrt)">

<%
    input_ctype = "int32_t" if i32_input else "uint64_t"
%>

${ntt_shared()}

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM FFP sh[1024];

    LOCAL_MEM FFP twd[1024];
    LOCAL_MEM FFP twd_sqrt[1024];

    VSIZE_T tid = virtual_local_id(1);
    VSIZE_T batch_id = virtual_group_id(0);

    for (int i = 0; i < 8; i++)
    {
        sh[tid + i * 128] = FFP((${input_ctype})${input_.load_combined_idx(slices)}(batch_id, tid + i * 128));
        twd[tid + i * 128] = FFP((uint64_t)${twd.load_idx}(tid + i * 128));
        twd_sqrt[tid + i * 128] = FFP((uint64_t)${twd_sqrt.load_idx}(tid + i * 128));
    }

    LOCAL_BARRIER;

    NTT1024<FFP>(sh, sh, sh, twd, twd_sqrt, tid >> 7 << 7);

    LOCAL_BARRIER;

    for (int i = 0; i < 8; i++)
    {
        ${output.store_combined_idx(slices)}(batch_id, tid + i * 128, sh[tid + i * 128].val());
    }
}
</%def>


<%def name="ntt1024_inv(kernel_declaration, output, input_, twd_inv, twd_sqrt_inv)">

<%
    output_ctype = "int32_t" if i32_output else "uint64_t"
%>

${ntt_shared()}

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM FFP sh[1024];
    LOCAL_MEM FFP twd_inv[1024];
    LOCAL_MEM FFP twd_sqrt_inv[1024];

    VSIZE_T tid = virtual_local_id(1);
    VSIZE_T batch_id = virtual_group_id(0);

    for (int i = 0; i < 8; i++)
    {
        sh[tid + i * 128] = FFP((uint64_t)${input_.load_combined_idx(slices)}(batch_id, tid + i * 128));
        twd_inv[tid + i * 128] = FFP((uint64_t)${twd_inv.load_idx}(tid + i * 128));
        twd_sqrt_inv[tid + i * 128] = FFP((uint64_t)${twd_sqrt_inv.load_idx}(tid + i * 128));
    }

    LOCAL_BARRIER;

    NTTInv1024<FFP>(sh, sh, sh, twd_inv, twd_sqrt_inv, tid >> 7 << 7);

    LOCAL_BARRIER;

    for (int i = 0; i < 8; i++)
    {
        ${output.store_combined_idx(slices)}(batch_id, tid + i * 128, (${output_ctype})(sh[tid + i * 128]));
    }
}
</%def>
