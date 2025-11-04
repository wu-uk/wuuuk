# 1 数论

## 1.1 快速幂

```cpp
int ksm(int x, int n = MOD - 2, int p = MOD) {
    int ans = 1;
    for (x %= p; n; n >>= 1, x = x * x % p)
        if (n & 1) ans = ans * x % p;
    return ans;
}
```

## 1.2 线性筛

预处理积性函数用，同时筛出素数。常见的有因数个数、因数和、莫比乌斯函数、欧拉函数等。

- 若 $p\nmid i$，则 $f_{p\cdot i} = f_p \cdot f_i$。
- 若 $p\mid i$，需要得到 $f_{p^k i} \rightarrow f_{p^{k+1}i}$ 的递推关系。

以下是求解因数个数以及因数和的模板。

```cpp
vector<int> primes, minp;
vector<int> cnt;           // cnt[i] = 指数 e（i 中最小质因子的幂）
vector<long long> pw;      // pw[i] = p^e（对应最小质因子部分）
vector<long long> dcnt;    // 约数个数 d(n)
vector<long long> dsum;    // 约数和 sigma(n)

void sieve(int n) {
    primes.reserve(2 * n / __lg(n));
    minp.assign(n + 1, 0);
    cnt.assign(n + 1, 0);
    pw.assign(n + 1, 0);
    dcnt.assign(n + 1, 0);
    dsum.assign(n + 1, 0);

    minp[1] = 1;
    cnt[1] = 0;
    pw[1] = 1;
    dcnt[1] = 1;   // d(1) = 1
    dsum[1] = 1;   // sigma(1) = 1

    for (int i = 2; i <= n; ++i) {
        if (!minp[i]) {
            primes.emplace_back(i);
            minp[i] = i;
            cnt[i] = 1;      
            pw[i] = i;       
            dcnt[i] = 2;      
            dsum[i] = 1 + (long long)i; 
        }
        for (int p : primes) {
            long long ip = 1LL * i * p;
            if (p > minp[i] || ip > n) break;
            minp[ip] = p;
            if (i % p == 0) {
                cnt[ip] = cnt[i] + 1;
                pw[ip]  = pw[i] * p;
                dcnt[ip] = dcnt[i] / (cnt[i] + 1) * (cnt[i] + 2);
                dsum[ip] = dsum[i] / (1 + pw[i]) * (1 + pw[ip]);
            } else {
                cnt[ip] = 1;
                pw[ip]  = p;
                dcnt[ip] = dcnt[i] * 2;  
                dsum[ip] = dsum[i] * (1 + (long long)p);
            }
        }
    }
}
```

## 1.3 欧拉函数

1到n中与n互素的整数个数，是积性函数。

### 1.3.1 单点查

用到公式 $\varphi(n)=n\prod_{i=1}^{m}(1-\frac{1}{p_i})$。m为质因数的个数。复杂度 $O(\sqrt{n})$。

```cpp
long long phi(long long n) {
    long long res = n;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) n /= i;
            res = res / i * (i - 1);
        }
    }
    if (n > 1) res = res / n * (n - 1);
    return res;
}
```

### 1.3.2 线性筛

用到性质 $\varphi(p^k) = p^k - p^{k-1}$。复杂度$O(n)$。

```cpp
vector<int> primes, minp, phi;
void sieve(int n) {
    primes.reserve(2 * n / __lg(n));
    minp.assign(n + 1, 0);
    phi.assign(n + 1, 0);
    minp[1] = phi[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!minp[i]) {
            primes.emplace_back(i);
            minp[i] = i;
            phi[i] = i - 1;
        }
        for (int p : primes) {
            if (p > minp[i] or i * p > n) break;
            minp[i * p] = p;
            if (i % p == 0) {
                phi[i * p] = phi[i] * p;
            } else {
                phi[i * p] = phi[i] * phi[p];
            }
        }
    }
}
```

### 1.3.3 欧拉反演

$$
n = \sum_{d\mid n} \varphi(d)
$$

$$
\gcd(a, b) = \sum_{d}[d\mid a][d\mid b]\varphi(d)
$$

## 1.4 莫比乌斯函数

$$
\mu(n) = 
\begin{cases} 
1 & \text{如果 } n = 1, \\
0 & \text{如果 } n \text{ 有一个平方的质因子}, \\
(-1)^k & \text{如果 } n \text{ 是 } k \text{ 个不同质数的乘积}.
\end{cases}
$$

### 1.4.1 线性筛

```cpp
void sieve(int n) {
    minp.assign(n + 1, 0);
    miu.assign(n + 1, 0);
    miu[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!minp[i]) {
            primes.emplace_back(minp[i] = i);
            miu[i] = -1;
        }
        for (int p : primes) {
            if (p * i > n or p > minp[i]) break;
            minp[i*p] = p;
            miu[i*p] = i % p ? -miu[i] : 0;
        }
    }
}
```

### 1.4.2 莫比乌斯反演

$$
\sum_{d|n} \mu(d) = [n = 1]
$$

$$
[\gcd(i, j) = 1] = \sum_{d|\gcd(i, j)} \mu(d)
$$

## 1.5 和式变换技巧

1. 替换条件式
   $$
   \sum_{i=1}^{n} \sum_{j=1}^{m} \sum_{d|\gcd(i,j)} d = \sum_{i=1}^{n} \sum_{j=1}^{m} \sum_{d=1}^{n} [d | i][d | j]d
   $$

2. 替换指标变量
   $$
   \sum_{i=1}^{n} \sum_{j=1}^{m} [\gcd(i,j) = k] = \sum_{ik=1}^{n} \sum_{jk=1}^{m} [\gcd(ik,jk) = k]
   $$

3. 交换求和次序
   $$
   \sum_{i=1}^{n} \sum_{j=1}^{m} A(i)B(j) = \sum_{j=1}^{m} \sum_{i=1}^{n} A(i)B(j)
   $$

4. 分离变量
   $$
   \sum_{i=1}^{n} \sum_{j=1}^{m} A(i)B(j) = \sum_{i=1}^{n} A(i) \sum_{j=1}^{m} B(j)
   $$

## 1.6 扩展欧几里得算法 （exgcd）

```cpp
// gcd(x, y) = a * x + b * y, return gcd(x, y)
int exgcd(int x, int y, int &a, int &b) {
    if (!y) {
        a = 1, b = 0;
        return x;
    }
    int d = exgcd(y, x % y, b, a);
    b -= a * (x / y);
    return d;
}
```

# 2 组合数学

## 2.1 组合数学常见性质

- $k *C^k_n=n*C^{k-1}_{n-1}$ ；
- $C_k^n*C_m^k=C_m^n*C_{m-n}^{m-k}$ ；
- $C_n^k+C_n^{k+1}=C_{n+1}^{k+1}$ ；
- $\sum_{i=0}^n C_n^i=2^n$ ；
- $\sum_{k=0}^n(-1)^k*C_n^k=0$ 。
- 二项式反演：$\left\{\begin{matrix} \displaystyle f_n=\sum_{i=0}^n{n\choose i}g_i\Leftrightarrow g_n=\sum_{i=0}^n(-1)^{n-i}{n\choose i}f_i \\ 
\displaystyle f_k=\sum_{i=k}^n{i\choose k}g_i\Leftrightarrow g_k=\sum_{i=k}^n(-1)^{i-k}{i\choose k}f_i \end{matrix}\right. $ ；
- $\displaystyle \sum_{i=1}^{n}i{n\choose i}=n * 2^{n-1}$ ；
- $\displaystyle \sum_{i=1}^{n}i^2{n\choose i}=n*(n+1)*2^{n-2}$ ；
- $\displaystyle \sum_{i=1}^{n}\dfrac{1}{i}{n\choose i}=\sum_{i=1}^{n}\dfrac{1}{i}$ ；
- $\displaystyle \sum_{i=0}^{n}{n\choose i}^2={2n\choose n}$ ；
- 拉格朗日恒等式：$\displaystyle \sum_{i=1}^{n}\sum_{j=i+1}^{n}(a_ib_j-a_jb_i)^2=(\sum_{i=1}^{n}a_i)^2(\sum_{i=1}^{n}b_i)^2-(\sum_{i=1}^{n}a_ib_i)^2$ 。

## 2.2 范德蒙德卷积公式

在数量为 $n+m$ 的堆中选 $k$ 个元素，和分别在数量为 $n、m$ 的堆中选 $i、k-i$ 个元素的方案数是相同的，即$\displaystyle{\sum_{i=0}^k\binom{n}{i}\binom{m}{k-i}=\binom{n+m}{k}}$ ；

变体：

- $\sum_{i=0}^k C_{i+n}^{i}=C_{k+n+1}^{k}$ ；
- $\sum_{i=0}^k C_{n}^{i}*C_m^i=\sum_{i=0}^k C_{n}^{i}*C_m^{m-i}=C_{n+m}^{n}$ 。

## 2.3 卡特兰数

Catalan 数列 $H_n$ 可以应用于以下问题：

1.  有 $2n$ 个人排成一行进入剧场。入场费 5 元。其中只有 $n$ 个人有一张 5 元钞票，另外 $n$ 人只有 10 元钞票，剧院无其它钞票，问有多少种方法使得只要有 10 元的人买票，售票处就有 5 元的钞票找零？
2.  有一个大小为 $n\times n$ 的方格图左下角为 $(0, 0)$ 右上角为 $(n, n)$，从左下角开始每次都只能向右或者向上走一单位，不走到对角线 $y=x$ 上方（但可以触碰）的情况下到达右上角有多少可能的路径？
3.  在圆上选择 $2n$ 个点，将这些点成对连接起来使得所得到的 $n$ 条线段不相交的方法数？
4.  对角线不相交的情况下，将一个凸多边形区域分成三角形区域的方法数？
5.  一个栈（无穷大）的进栈序列为 $1,2,3, \cdots ,n$ 有多少个不同的出栈序列？
6.  $n$ 个结点可构造多少个不同的二叉树？
7.  由 $n$ 个 $+1$ 和 $n$ 个 $-1$ 组成的 $2n$ 个数 $a_1,a_2, \cdots ,a_{2n}$，其部分和满足 $a_1+a_2+ \cdots +a_k \geq 0~(k=1,2,3, \cdots ,2n)$，有多少个满足条件的数列？

其对应的序列为：

| $H_0$ | $H_1$ | $H_2$ | $H_3$ | $H_4$ | $H_5$ | $H_6$ | ...  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: |
|   1   |   1   |   2   |   5   |  14   |  42   |  132  | ...  |

### 2.3.1 递推式

该递推关系的解为：


$$
H_n = \frac{\binom{2n}{n}}{n+1}(n \geq 2, n \in \mathbf{N_{+}})
$$


关于 Catalan 数的常见公式：

$$
H_n = \begin{cases}
    \sum_{i=1}^{n} H_{i-1} H_{n-i} & n \geq 2, n \in \mathbf{N_{+}}\\
    1 & n = 0, 1
\end{cases}
$$

$$
H_n = \frac{H_{n-1} (4n-2)}{n+1}
$$

$$
H_n = \binom{2n}{n} - \binom{2n}{n-1}
$$

### 2.3.2 路径计数问题

非降路径是指只能向上或向右走的路径。

1.  从 $(0,0)$ 到 $(m,n)$ 的非降路径数等于 $m$ 个 $x$ 和 $n$ 个 $y$ 的排列数，即 $\dbinom{n + m}{m}$。

2.  从 $(0,0)$ 到 $(n,n)$ 的除端点外不接触直线 $y=x$ 的非降路径数：

    先考虑 $y=x$ 下方的路径，都是从 $(0, 0)$ 出发，经过 $(1, 0)$ 及 $(n, n-1)$ 到 $(n,n)$，可以看做是 $(1,0)$ 到 $(n,n-1)$ 不接触 $y=x$ 的非降路径数。

    所有的的非降路径有 $\dbinom{2n-2}{n-1}$ 条。对于这里面任意一条接触了 $y=x$ 的路径，可以把它最后离开这条线的点到 $(1,0)$ 之间的部分关于 $y=x$ 对称变换，就得到从 $(0,1)$ 到 $(n,n-1)$ 的一条非降路径。反之也成立。从而 $y=x$ 下方的非降路径数是 $\dbinom{2n-2}{n-1} - \dbinom{2n-2}{n}$。根据对称性可知所求答案为 $2\dbinom{2n-2}{n-1} - 2\dbinom{2n-2}{n}$。

3.  从 $(0,0)$ 到 $(n,n)$ 的除端点外不穿过直线 $y=x$ 的非降路径数：

    用类似的方法可以得到：$\dfrac{2}{n+1}\dbinom{2n}{n}$​

## 2.4 斐波那契数列

通项公式：$F_n=\dfrac{1}{\sqrt 5}*  \Big[ \Big( \dfrac{1+\sqrt 5}{2} \Big)^n - \Big( \dfrac{1-\sqrt 5}{2} \Big)^n \Big]$ 。

直接结论：

- 卡西尼性质：$F_{n-1} * F_{n+1}-F_n^2=(-1)^n$ ；
- $F_{n}^2+F_{n+1}^2=F_{2n+1}$ ；
- $F_{n+1}^2-F_{n-1}^2=F_{2n}$ （由上一条写两遍相减得到）；
- 若存在序列 $a_0=1,a_n=a_{n-1}+a_{n-3}+a_{n-5}+...(n\ge 1)$ 则 $a_n=F_n(n\ge 1)$ ；
- 齐肯多夫定理：任何正整数都可以表示成若干个不连续的斐波那契数（ $F_2$ 开始）可以用贪心实现。

求和公式结论：

- 奇数项求和：$F_1+F_3+F_5+...+F_{2n-1}=F_{2n}$ ；
- 偶数项求和：$F_2+F_4+F_6+...+F_{2n}=F_{2n+1}-1$ ；
- 平方和：$F_1^2+F_2^2+F_3^2+...+F_n^2=F_n*F_{n+1}$ ；
- $F_1+2F_2+3F_3+...+nF_n=nF_{n+2}-F_{n+3}+2$ ；
- $-F_1+F_2-F_3+...+(-1)^nF_n=(-1)^n(F_{n+1}-F_n)+1$ ；
- $F_{2n-2m-2}(F_{2n}+F_{2n+2})=F_{2m+2}+F_{4n-2m}$ 。

数论结论：

- $F_a \mid F_b \Leftrightarrow a \mid b$ ；
- $\gcd(F_a,F_b)=F_{\gcd(a,b)}$ ；
- 当 $p$ 为 $5k\pm 1$ 型素数时，$\begin{cases} F_{p-1}\equiv 0\pmod p \\ F_p\equiv 1\pmod p \\ F_{p+1}\equiv 1\pmod p \end{cases}$ ；
- 当 $p$ 为 $5k\pm 2$ 型素数时，$\begin{cases} F_{p-1}\equiv 1\pmod p \\ F_p\equiv -1\pmod p \\ F_{p+1}\equiv 0\pmod p \end{cases}$ ；
- $F(n)\%m$ 的周期 $\le 6m$ （ $m=2\times 5^k$ 时取到等号）；
- 既是斐波那契数又是平方数的有且仅有 $1,144$ 。

快速倍增法（比矩阵乘常数小）：
```cpp
pair<int, int> fib(int n) { // O(logn)返回F(n),F(n+1)
  if (n == 0) return {0, 1};
  auto p = fib(n >> 1);
  int c = p.first * (2 * p.second - p.first);
  int d = p.first * p.first + p.second * p.second;
  if (n & 1)
    return {d, c + d};
  else
    return {c, d};
}
```

---

## 2.5 约瑟夫问题

> n 个人标号 $0,1,\cdots, n-1$。逆时针站一圈，从 $0$ 号开始，每一次从当前的人逆时针数 $k$ 个，然后让这个人出局。问最后剩下的人是谁。

### 2.5.1 线性算法

设 $J_{n,k}$ 表示规模分别为 $n,k$ 的约瑟夫问题的答案。我们有如下递归式

$$
J_{n,k}=(J_{n-1,k}+k)\bmod n
$$

这个也很好推。你从 $0$ 开始数 $k$ 个，让第 $k-1$ 个人出局后剩下 $n-1$ 个人，你计算出在 $n-1$ 个人中选的答案后，再加一个相对位移 $k$ 得到真正的答案。这个算法的复杂度显然是 $\Theta (n)$ 的。

```cpp
int josephus(int n, int k) {
  int res = 0;
  for (int i = 1; i <= n; ++i) res = (res + k) % i;
  return res;
}
```

### 2.5.2 对数算法

对于 $k$ 较小 $n$ 较大的情况，本题还有一种复杂度为 $\Theta (k\log n)$ 的算法。

考虑到我们每次走 $k$ 个删一个，那么在一圈以内我们可以删掉 $\left\lfloor\frac{n}{k}\right\rfloor$ 个，然后剩下了 $n-\left\lfloor\frac{n}{k}\right\rfloor$ 个人。这时我们在第 $\left\lfloor\frac{n}{k}\right\rfloor\cdot k$ 个人的位置上。而你发现它等于 $n-n\bmod k$。于是我们继续递归处理，算完后还原它的相对位置。还原相对位置的依据是：每次做一次删除都会把数到的第 $k$ 个人删除，他们的编号被之后的人逐个继承，也即用 $n-\left\lfloor\frac{n}{k}\right\rfloor$ 人环算时每 $k$ 个人即有 $1$ 个人的位置失算，因此在得数小于 $0$ 时，用还没有被删去 $k$ 倍数编号的 $n$ 人环的 的 $n$ 求模，在得数大于等于 $0$ 时，即可以直接乘 $\frac{k}{k-1}$, 于是得到如下的算法：

```cpp
int josephus(int n, int k) {
  if (n == 1) return 0;
  if (k == 1) return n - 1;
  if (k > n) return (josephus(n - 1, k) + k) % n;  // 线性算法
  int res = josephus(n - n / k, k);
  res -= n % k;
  if (res < 0) res += n;  // mod n
  else res += res / (k - 1);  // 还原位置
  return res;
}
```

## 2.6 格雷码

```cpp
int g(int n) { return n ^ (n >> 1); }
```

### 2.6.1 通过格雷码构造原数（逆变换）

接下来我们考虑格雷码的逆变换，即给你一个格雷码 $g$，要求你找到原数 $n$。我们考虑从二进制最高位遍历到最低位（最低位下标为 $1$，即个位；最高位下标为 $k$）。则 $n$ 的二进制第 $i$ 位与 $g$ 的二进制第 $i$ 位 $g_i$ 的关系如下：

$$
\begin{aligned}
n_k &= g_k \\
n_{k-1} &= g_{k-1} \oplus n_k &&= g_k \oplus g_{k-1} \\
n_{k-2} &= g_{k-2} \oplus n_{k-1} &&= g_k \oplus g_{k-1} \oplus g_{k-2} \\
n_{k-3} &= g_{k-3} \oplus n_{k-2} &&= g_k \oplus g_{k-1} \oplus g_{k-2} \oplus g_{k-3} \\
&\vdots\\
n_{k-i} &=\displaystyle\bigoplus_{j=0}^ig_{k-j}
\end{aligned}
$$

```cpp
int rev_g(int g) {
  int n = 0;
  for (; g; g >>= 1) n ^= g;
  return n;
}
```

## 2.7 错位排列

### 2.7.1 定义

错位排列（derangement）是没有任何元素出现在其有序位置的排列。即，对于 $1\sim n$ 的排列 $P$，如果满足 $P_i\neq i$，则称 $P$ 是 $n$ 的错位排列。

例如，三元错位排列有 $\{2,3,1\}$ 和 $\{3,1,2\}$。四元错位排列有 $\{2,1,4,3\}$、$\{2,3,4,1\}$、$\{2,4,1,3\}$、$\{3,1,4,2\}$、$\{3,4,1,2\}$、$\{3,4,2,1\}$、$\{4,1,2,3\}$、$\{4,3,1,2\}$ 和 $\{4,3,2,1\}$。错位排列是没有不动点的排列，即没有长度为 1 的循环。

错位排列数列的前几项为 $0,1,2,9,44,265$（[OEIS A000166](http://oeis.org/A000166)）。

### 2.7.2 递推的计算

错位排列数满足递推关系：

$$
D_n=(n-1)(D_{n-1}+D_{n-2})
$$

这里也给出另一个递推关系：

$$
D_n=nD_{n-1}+{(-1)}^n
$$

### 2.7.3 其他关系

错位排列数有一个简单的取整表达式，增长速度与阶乘仅相差常数：

$$
D_n=\begin{cases}
    \left\lceil\frac{n!}{\mathrm{e}}\right\rceil, & \text{if }n\text{ is even}, \\
    \left\lfloor\frac{n!}{\mathrm{e}}\right\rfloor,            & \text{if }n\text{ is odd}.
\end{cases}
$$

随着元素数量的增加，形成错位排列的概率 P 接近：

$$
P=\lim_{n\to\infty}\frac{D_n}{n!}=\frac{1}{\mathrm{e}}
$$

## 2.8 康托展开

$O(n\log{n})$ 求出一个排列在所有长度为 $n$ 的排列中的字典序排名。

```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
constexpr int MOD = 998244353;
constexpr int N = 1e6 + 5;
int n, x, d[N], fac[N], ans;
void add(int i, int x) {
    for (; i <= n; i += (i & -i)) d[i] += x;
}
int query(int i, int s = 0) {
    for (; i > 0; i -= (i & -i)) s += d[i];
    return s;
}
signed main() {
    cin.tie(nullptr)->sync_with_stdio(false);
    cin >> n;
    fac[0] = 1;
    for (int i = 1; i <= n; ++i) {
        d[i] = (i & -i);                 // O(n) 建树
        fac[i] = (fac[i - 1] * i) % MOD;  // 预处理阶乘
    }
    for (int i = 1; i <= n; ++i) {
        cin >> x;
        add(x, -1);
        ans = (ans + query(x) * fac[n - i] % MOD) % MOD;
    }
    cout << ans + 1 << '\n';
}
```

## 2.9 逆康托展开

由于 $n$ 不会超过20（否则没法处理阶乘），这里提供暴力做法，复杂度 $O(n^2)$。

 ```cpp
 #define int long long
 constexpr int N = 21;
 int fac[N]; // 预处理一下阶乘表
 vector<int> get_inv_cantor(int n, int k) { // k从0开始
     vector<int> perm;
     vector<bool> used(n + 1, false);
     for (int i = n; i >= 1; --i) {
         int order = k / fac[i - 1];
         k %= fac[i - 1];
         for (int j = 1; j <= n; ++j) {
             if (used[j]) continue;
             if (order == 0) {
                 perm.push_back(j);
                 used[j] = true;
                 break;
             }
             order--;
         }
     }
     return perm;
 }
 ```

## 2.10 第二类斯特林数（Stirling Number）

**第二类斯特林数**（斯特林子集数）$\begin{Bmatrix}n\\ k\end{Bmatrix}$，也可记做 $S(n,k)$，表示将 $n$ 个两两不同的元素，划分为 $k$ 个互不区分的非空子集的方案数。

### 2.10.1 递推式

$$
\begin{Bmatrix}n\\ k\end{Bmatrix}=\begin{Bmatrix}n-1\\ k-1\end{Bmatrix}+k\begin{Bmatrix}n-1\\ k\end{Bmatrix}
$$

边界是 $\begin{Bmatrix}n\\ 0\end{Bmatrix}=[n=0]$。

### 2.10.2 通项公式

$$
\begin{Bmatrix}n\\m\end{Bmatrix}=\sum\limits_{i=0}^m\dfrac{(-1)^{m-i}i^n}{i!(m-i)!}
$$

### 2.10.3 同一行第二类斯特林数的计算

「同一行」的第二类斯特林数指的是，有着不同的 $i$，相同的 $n$ 的一系列 $\begin{Bmatrix}n\\i\end{Bmatrix}$。求出同一行的所有第二类斯特林数，就是对 $i=0..n$ 求出了将 $n$ 个不同元素划分为 $i$ 个非空集的方案数。

根据上面给出的通项公式，卷积计算即可。该做法的时间复杂度为 $O(n \log n)$。

# 3 线性代数

## 3.1 高斯消元

求解线性方程组，`a[i][0] ~ a[i][n-1]` 为系数，`a[i][n]` 为等号右边的常数，最后结果保存在`a[i][n]`中。

```cpp
const int N = 110;
const double eps = 1e-8;
LL n;
double a[N][N];
LL gauss(){
    LL c, r;
    for (c = 0, r = 0; c < n; c ++ ){
        LL t = r;
        for (int i = r; i < n; i ++ )    //找到绝对值最大的行 
            if (fabs(a[i][c]) > fabs(a[t][c]))
                t = i;
        if (fabs(a[t][c]) < eps) continue;
        for (int j = c; j < n + 1; j ++ ) swap(a[t][j], a[r][j]);    //将绝对值最大的一行换到最顶端
        for (int j = n; j >= c; j -- ) a[r][j] /= a[r][c];    //将当前行首位变成 1
        for (int i = r + 1; i < n; i ++ )    //将下面列消成 0 
            if (fabs(a[i][c]) > eps)
                for (int j = n; j >= c; j -- )
                    a[i][j] -= a[r][j] * a[i][c];
        r ++ ;
    }
    if (r < n){
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][n]) > eps)
                return 2;
        return 1;
    }
    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            a[i][n] -= a[i][j] * a[j][n];
    return 0;
}
int main(){
    cin >> n;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n + 1; j ++ )
            cin >> a[i][j];
    LL t = gauss();
    if (t == 0){
        for (int i = 0; i < n; i ++ ){
            if (fabs(a[i][n]) < eps) a[i][n] = abs(a[i][n]);
            printf("%.2lf\n", a[i][n]);
        }
    }
    else if (t == 1) cout << "Infinite group solutions\n";
    else cout << "No solution\n";
    return 0;
}
```

## 3.2 线性基

```cpp
constexpr int L = 60;
ll basis[L];
int cnt = 0;
bool insert(ll x) {
    for (int k = L - 1; k >= 0; k--) {
        if (((x >> k) & 1LL) == 0) continue;
        if (basis[k] == 0) {
            basis[k] = x;
            cnt++;
            return true;
        }
        x ^= basis[k];
    }
    return false;
}
```

# 4 多项式

## 4.1 NTT

init需要传入卷积的最长长度，N需要比大于这个长度的2的幂更大。

```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
constexpr int RN = 1e6 + 5;
constexpr int N = (1 << (__lg(2 * RN) + 1)) + 5;
constexpr int p = 998244353;
inline int add(const int &x, const int &y) { return x + y >= p ? x + y - p : x + y; }
inline int dec(const int &x, const int &y) { return x < y ? x - y + p : x - y; }

inline int power(int a, int t) {
    int res = 1;
    while (t) {
        if (t & 1) res = res * a % p;
        a = a * a % p;
        t >>= 1;
    }
    return res;
}

int siz;
int rev[N], rt[N], inv[N], fac[N], ifac[N];

void init(int n) {  // 传入最长卷积长度
    int lim = 1;
    while (lim < n) lim <<= 1, ++siz;
    for (int i = 0; i != lim; ++i) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (siz - 1));
    int w = power(3, (p - 1) >> siz);
    fac[0] = ifac[0] = rt[lim >> 1] = 1;
    for (int i = (lim >> 1) + 1; i != lim; ++i) rt[i] = rt[i - 1] * w % p;
    for (int i = (lim >> 1) - 1; i; --i) rt[i] = rt[i << 1];

    for (int i = 1; i <= n; ++i) fac[i] = fac[i - 1] * i % p;
    ifac[n] = power(fac[n], p - 2);
    for (int i = n - 1; i; --i) ifac[i] = ifac[i + 1] * (i + 1) % p;
    for (int i = 1; i <= n; ++i) inv[i] = fac[i - 1] * ifac[i] % p;
}

inline void dft(int *f, int n) {
    static unsigned long long a[N];
    int x, shift = siz - __builtin_ctz(n);
    for (int i = 0; i != n; ++i) a[rev[i] >> shift] = f[i];
    for (int mid = 1; mid != n; mid <<= 1)
        for (int j = 0; j != n; j += (mid << 1))
            for (int k = 0; k != mid; ++k) {
                x = a[j | k | mid] * rt[mid | k] % p;
                a[j | k | mid] = a[j | k] + p - x;
                a[j | k] += x;
            }
    for (int i = 0; i != n; ++i) f[i] = a[i] % p;
}

inline void idft(int *f, int n) {
    reverse(f + 1, f + n);
    dft(f, n);
    int x = p - ((p - 1) >> __builtin_ctz(n));
    for (int i = 0; i != n; ++i) f[i] = f[i] * x % p;
}

inline void conv(int *f, int fn, int *g, int gn, int *r) {
    int len = fn + gn - 1, lim = 1;
    while (lim < len) lim <<= 1;
    dft(f, lim);
    dft(g, lim);
    for (int i = 0; i < lim; ++i) r[i] = f[i] * g[i] % p;
    idft(r, lim);
}

inline int getlen(int n) { return 1 << (32 - __builtin_clz(n)); }

inline void inverse(const int *f, int n, int *r) {
    static int g[N], h[N], st[30];
    memset(g, 0, getlen(n << 1) * sizeof(int));
    int lim = 1, top = 0;
    while (n) {
        st[++top] = n;
        n >>= 1;
    }
    g[0] = 1;
    while (top--) {
        n = st[top + 1];
        while (lim <= (n << 1)) lim <<= 1;
        memcpy(h, f, (n + 1) * sizeof(int));
        memset(h + n + 1, 0, (lim - n) * sizeof(int));
        dft(g, lim), dft(h, lim);
        for (int i = 0; i != lim; ++i) g[i] = g[i] * (2 - g[i] * h[i] % p + p) % p;
        idft(g, lim);
        memset(g + n + 1, 0, (lim - n) * sizeof(int));
    }
    memcpy(r, g, (n + 1) * sizeof(int));
}

inline void log(const int *f, int n, int *r) {
    static int g[N], h[N];
    inverse(f, n, g);
    for (int i = 0; i != n; ++i) h[i] = f[i + 1] * (i + 1) % p;
    h[n] = 0;
    int lim = getlen(n << 1);
    memset(g + n + 1, 0, (lim - n) * sizeof(int));
    memset(h + n + 1, 0, (lim - n) * sizeof(int));
    dft(g, lim), dft(h, lim);
    for (int i = 0; i != lim; ++i) g[i] = g[i] * h[i] % p;
    idft(g, lim);
    for (int i = 1; i <= n; ++i) r[i] = g[i - 1] * inv[i] % p;
    r[0] = 0;
}

inline void exp(const int *f, int n, int *r) {
    static int g[N], h[N], st[30];
    memset(g, 0, getlen(n << 1) * sizeof(int));
    int lim = 1, top = 0;
    while (n) {
        st[++top] = n;
        n >>= 1;
    }
    g[0] = 1;
    while (top--) {
        n = st[top + 1];
        while (lim <= (n << 1)) lim <<= 1;
        memcpy(h, g, (n + 1) * sizeof(int));
        memset(h + n + 1, 0, (lim - n) * sizeof(int));
        log(g, n, g);
        for (int i = 0; i <= n; ++i) g[i] = dec(f[i], g[i]);
        g[0] = add(g[0], 1);
        dft(g, lim), dft(h, lim);
        for (int i = 0; i != lim; ++i) g[i] = g[i] * h[i] % p;
        idft(g, lim);
        memset(g + n + 1, 0, (lim - n) * sizeof(int));
    }
    memcpy(r, g, (n + 1) * sizeof(int));
}
```

## 4.2 FFT

三次变两次卷积。

```cpp
#include <bits/stdc++.h
using namespace std;
using cp = complex<double>;
using vp = vector<cp>;
using vi = vector<int>;
vi rev;    
void init_rev(int limit) {
    rev.resize(limit);
    for (int i = 0; i < limit; ++i) rev[i] = (rev[i / 2] / 2 + (i % 2) * limit / 2);
}
void FFT(vp &x, int limit, bool inv = false) {
    for (int i = 0; i < limit; ++i)
        if (i < rev[i]) swap(x[i], x[rev[i]]);
    for (int len = 1; len < limit; len <<= 1) {
        cp wn(cos(PI / len), (-2 * inv + 1) * sin(PI / len));
        for (int i = 0; i < limit; i += 2 * len) {
            cp w(1);
            for (int j = i; j < i + len; j++, w *= wn) {
                cp u = x[j], v = w * x[j + len];
                x[j] = u + v, x[j + len] = u - v;
            }
        }
    }
    if (!inv) return;
    for (auto &i : x) i /= limit;
}
vi operator*(const vi &a, const vi &b) {
    int len = a.size() + b.size() - 1;
    int limit = 1LL << __lg(len);
    if (limit < len) limit <<= 1;
    init_rev(limit);
    vp c(limit);
    for (size_t i = 0; i < limit; i++) {
        c[i] = (double)(i<a.size() ? a[i] : 0LL) + I * (double)(i<b.size() ? b[i] : 0LL);
    }
    FFT(c, limit);
    for (int i = 0; i < limit; ++i) c[i] = c[i] * c[i];
    FFT(c, limit, true);
    vi res(len);
    for (size_t i = 0; i < len; i++) {
        res[i] = (int)(0.5 * c[i].imag() + 0.5);
    }
    return res;
}
```

# 5 几何（by WIDA）

## 5.1 库实数类实现（双精度）

```c++
using Real = int;
using Point = complex<Real>;
 
Real cross(const Point &a, const Point &b) {
    return (conj(a) * b).imag();
} 
Real dot(const Point &a, const Point &b) {
    return (conj(a) * b).real();
}
```

## 5.2 平面几何必要初始化

### 5.2.1 字符串读入浮点数

```c++
const int Knum = 4;
int read(int k = Knum) {
    string s;
    cin >> s;
    
    int num = 0;
    int it = s.find('.');
    if (it != -1) { // 存在小数点
        num = s.size() - it - 1; // 计算小数位数
        s.erase(s.begin() + it); // 删除小数点
    }
    for (int i = 1; i <= k - num; i++) { // 补全小数位数
        s += '0';
    }
    return stoi(s); 
}
```

### 5.2.2 预置函数

```c++
using ld = long double;
const ld PI = acos(-1);
const ld EPS = 1e-7;
const ld INF = numeric_limits<ld>::max();
#define cc(x) cout << fixed << setprecision(x);

ld fgcd(ld x, ld y) { // 实数域gcd
    return abs(y) < EPS ? abs(x) : fgcd(y, fmod(x, y));
}
template<class T, class S> bool equal(T x, S y) {
    return -EPS < x - y && x - y < EPS;
}
template<class T> int sign(T x) {
    if (-EPS < x && x < EPS) return 0;
    return x < 0 ? -1 : 1;
}
```

### 5.2.3 点线封装

```c++
template<class T> struct Point { // 在C++17下使用 emplace_back 绑定可能会导致CE！
    T x, y;
    Point(T x_ = 0, T y_ = 0) : x(x_), y(y_) {} // 初始化
    template<class U> operator Point<U>() { // 自动类型匹配
        return Point<U>(U(x), U(y));
    }
    Point &operator+=(Point p) & { return x += p.x, y += p.y, *this; }
    Point &operator+=(T t) & { return x += t, y += t, *this; }
    Point &operator-=(Point p) & { return x -= p.x, y -= p.y, *this; }
    Point &operator-=(T t) & { return x -= t, y -= t, *this; }
    Point &operator*=(T t) & { return x *= t, y *= t, *this; }
    Point &operator/=(T t) & { return x /= t, y /= t, *this; }
    Point operator-() const { return Point(-x, -y); }
    friend Point operator+(Point a, Point b) { return a += b; }
    friend Point operator+(Point a, T b) { return a += b; }
    friend Point operator-(Point a, Point b) { return a -= b; }
    friend Point operator-(Point a, T b) { return a -= b; }
    friend Point operator*(Point a, T b) { return a *= b; }
    friend Point operator*(T a, Point b) { return b *= a; }
    friend Point operator/(Point a, T b) { return a /= b; }
    friend bool operator<(Point a, Point b) {
        return equal(a.x, b.x) ? a.y < b.y - EPS : a.x < b.x - EPS;
    }
    friend bool operator>(Point a, Point b) { return b < a; }
    friend bool operator==(Point a, Point b) { return !(a < b) && !(b < a); }
    friend bool operator!=(Point a, Point b) { return a < b || b < a; }
    friend auto &operator>>(istream &is, Point &p) {
        return is >> p.x >> p.y;
    }
    friend auto &operator<<(ostream &os, Point p) {
        return os << "(" << p.x << ", " << p.y << ")";
    }
};
template<class T> struct Line {
    Point<T> a, b;
    Line(Point<T> a_ = Point<T>(), Point<T> b_ = Point<T>()) : a(a_), b(b_) {}
    template<class U> operator Line<U>() { // 自动类型匹配
        return Line<U>(Point<U>(a), Point<U>(b));
    }
    friend auto &operator<<(ostream &os, Line l) {
        return os << "<" << l.a << ", " << l.b << ">";
    }
};
```

### 5.2.4 叉乘

定义公式 $a\times b=|a||b|\sin \theta$。

```c++
template<class T> T cross(Point<T> a, Point<T> b) { // 叉乘
    return a.x * b.y - a.y * b.x;
}
template<class T> T cross(Point<T> p1, Point<T> p2, Point<T> p0) { // 叉乘 (p1 - p0) x (p2 - p0);
    return cross(p1 - p0, p2 - p0);
}
```

### 5.2.5 点乘

定义公式 $a\times b=|a||b|\cos \theta$。

```c++
template<class T> T dot(Point<T> a, Point<T> b) { // 点乘
    return a.x * b.x + a.y * b.y;
}
template<class T> T dot(Point<T> p1, Point<T> p2, Point<T> p0) { // 点乘 (p1 - p0) * (p2 - p0);
    return dot(p1 - p0, p2 - p0);
}
```

### 5.2.6 欧几里得距离公式

最常用的距离公式。**需要注意**，开根号会丢失精度，如无强制要求，先不要开根号，留到最后一步一起开。

```c++
template <class T> ld dis(T x1, T y1, T x2, T y2) {
    ld val = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    return sqrt(val);
}
template <class T> ld dis(Point<T> a, Point<T> b) {
    return dis(a.x, a.y, b.x, b.y);
}
```

### 5.2.7 曼哈顿距离公式

```c++
template <class T> T dis1(Point<T> p1, Point<T> p2) { // 曼哈顿距离公式
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}
```

### 5.2.8 将向量转换为单位向量

```c++
Point<ld> standardize(Point<ld> vec) { // 转换为单位向量
    return vec / sqrt(vec.x * vec.x + vec.y * vec.y);
}
```

### 5.2.9 向量旋转

将当前向量移动至原点后顺时针旋转 $90^{\circ}$ ，即获取垂直于当前向量的、起点为原点的向量。在计算垂线时非常有用。例如，要想获取点 $a$ 绕点 $o$ 顺时针旋转 $90^{\circ}$ 后的点，可以这样书写代码：`auto ans = o + rotate(o, a);` ；如果是逆时针旋转，那么只需更改符号即可：`auto ans = o - rotate(o, a);` 。

```c++
template<class T> Point<T> rotate(Point<T> p1, Point<T> p2) { // 旋转
    Point<T> vec = p1 - p2;
    return {-vec.y, vec.x};
}
```

## 5.3 平面角度与弧度

### 5.3.1 弧度角度相互转换

```c++
ld toDeg(ld x) { // 弧度转角度
    return x * 180 / PI;
}
ld toArc(ld x) { // 角度转弧度
    return PI / 180 * x;
}
```

### 5.3.2 正弦定理

$\dfrac{a}{\sin A}=\dfrac{b}{\sin B}=\dfrac{c}{\sin C}=2R$ ，其中 $R$ 为三角形外接圆半径；

### 5.3.3 余弦定理（已知三角形三边，求角）

$\cos C=\dfrac{a^2+b^2-c^2}{2ab},\cos B=\dfrac{a^2+c^2-b^2}{2ac},\cos A=\dfrac{b^2+c^2-a^2}{2bc}$。可以借此推导出三角形面积公式 $S_{\triangle ABC}=\dfrac{ab\cdot\sin C}{2}=\dfrac{bc\cdot\sin A}{2}=\dfrac{ac\cdot\sin B}{2}$。

注意，计算格式是：由 $b,c,a$ 三边求 $\angle A$；由 $a, c, b$ 三边求 $\angle B$；由 $a, b, c$ 三边求 $\angle C$。

```c++
ld angle(ld a, ld b, ld c) { // 余弦定理
    ld val = acos((a * a + b * b - c * c) / (2.0 * a * b)); // 计算弧度
    return val;
}
```

### 5.3.4 求两向量的夹角

能够计算 $[0^{\circ},180^{\circ}]$ 区间的角度。

```c++
ld angle(Point<ld> a, Point<ld> b) {
    ld val = abs(cross(a, b));
    return abs(atan2(val, a.x * b.x + a.y * b.y));
}
```

### 5.3.5 向量旋转任意角度

逆时针旋转，转换公式：$\left\{\begin{matrix}
x'=x\cos \theta-y\sin \theta \\ 
y'=x\sin \theta+y\cos \theta
\end{matrix}\right.$

```c++
Point<ld> rotate(Point<ld> p, ld rad) {
    return {p.x * cos(rad) - p.y * sin(rad), p.x * sin(rad) + p.y * cos(rad)};
}
```

### 5.3.6 点绕点旋转任意角度

逆时针旋转，转换公式：$\left\{\begin{matrix}
x'=(x_0-x_1)\cos\theta+(y_0-y_1)\sin\theta+x_1 \\ 
y'=(x_1-x_0)\sin\theta+(y_0-y_1)\cos\theta+y_1
\end{matrix}\right.$

```c++
Point<ld> rotate(Point<ld> a, Point<ld> b, ld rad) {
    ld x = (a.x - b.x) * cos(rad) + (a.y - b.y) * sin(rad) + b.x;
    ld y = (b.x - a.x) * sin(rad) + (a.y - b.y) * cos(rad) + b.y;
    return {x, y};
}
```

## 5.4 平面点线相关

### 5.4.1 点是否在直线上（三点是否共线）

```c++
template<class T> bool onLine(Point<T> a, Point<T> b, Point<T> c) {
    return sign(cross(b, a, c)) == 0;
}
template<class T> bool onLine(Point<T> p, Line<T> l) {
    return onLine(p, l.a, l.b);
}
```

### 5.4.2 点是否在向量（直线）左侧

**需要注意**，向量的方向会影响答案；点在向量上时不视为在左侧。

```c++
template<class T> bool pointOnLineLeft(Pt p, Lt l) {
    return cross(l.b, p, l.a) > 0;
}
```

### 5.4.3 两点是否在直线同侧/异侧

```c++
template<class T> bool pointOnLineSide(Pt p1, Pt p2, Lt vec) {
    T val = cross(p1, vec.a, vec.b) * cross(p2, vec.a, vec.b);
    return sign(val) == 1;
}
template<class T> bool pointNotOnLineSide(Pt p1, Pt p2, Lt vec) {
    T val = cross(p1, vec.a, vec.b) * cross(p2, vec.a, vec.b);
    return sign(val) == -1;
}
```

### 5.4.4 两直线相交交点

在使用前需要先判断直线是否平行。

```c++
Pd lineIntersection(Ld l1, Ld l2) {
    ld val = cross(l2.b - l2.a, l1.a - l2.a) / cross(l2.b - l2.a, l1.a - l1.b);
    return l1.a + (l1.b - l1.a) * val;
}
```

### 5.4.5 两直线是否平行/垂直/相同

```c++
template<class T> bool lineParallel(Lt p1, Lt p2) {
    return sign(cross(p1.a - p1.b, p2.a - p2.b)) == 0;
}
template<class T> bool lineVertical(Lt p1, Lt p2) {
    return sign(dot(p1.a - p1.b, p2.a - p2.b)) == 0;
}
template<class T> bool same(Line<T> l1, Line<T> l2) {
    return lineParallel(Line{l1.a, l2.b}, {l1.b, l2.a}) &&
           lineParallel(Line{l1.a, l2.a}, {l1.b, l2.b}) && lineParallel(l1, l2);
}
```

### 5.4.6 点到直线的最近距离与最近点

```c++
pair<Pd, ld> pointToLine(Pd p, Ld l) {
    Pd ans = lineIntersection({p, p + rotate(l.a, l.b)}, l);
    return {ans, dis(p, ans)};
}
```

如果只需要计算最近距离，下方的写法可以减少书写的代码量，效果一致。

```c++
template<class T> ld disPointToLine(Pt p, Lt l) {
    ld ans = cross(p, l.a, l.b);
    return abs(ans) / dis(l.a, l.b); // 面积除以底边长
}
```

### 5.4.7 点是否在线段上

```c++
template<class T> bool pointOnSegment(Pt p, Lt l) { // 端点也算作在直线上
    return sign(cross(p, l.a, l.b)) == 0 && min(l.a.x, l.b.x) <= p.x && p.x <= max(l.a.x, l.b.x) &&
           min(l.a.y, l.b.y) <= p.y && p.y <= max(l.a.y, l.b.y);
}
template<class T> bool pointOnSegment(Pt p, Lt l) { // 端点不算
    return pointOnSegment(p, l) && min(l.a.x, l.b.x) < p.x && p.x < max(l.a.x, l.b.x) &&
           min(l.a.y, l.b.y) < p.y && p.y < max(l.a.y, l.b.y);
}
```

### 5.4.8 点到线段的最近距离与最近点

```c++
pair<Pd, ld> pointToSegment(Pd p, Ld l) {
    if (sign(dot(p, l.b, l.a)) == -1) { // 特判到两端点的距离
        return {l.a, dis(p, l.a)};
    } else if (sign(dot(p, l.a, l.b)) == -1) {
        return {l.b, dis(p, l.b)};
    }
    return pointToLine(p, l);
}
```

### 5.4.9 点在直线上的投影点（垂足）

```c++
Pd project(Pd p, Ld l) { // 投影
    Pd vec = l.b - l.a;
    ld r = dot(vec, p - l.a) / (vec.x * vec.x + vec.y * vec.y);
    return l.a + vec * r;
}
```

### 5.4.10 线段的中垂线

```c++
template<class T> Lt midSegment(Lt l) {
    Pt mid = (l.a + l.b) / 2; // 线段中点
    return {mid, mid + rotate(l.a, l.b)};
}
```

### 5.4.11 两线段是否相交及交点

该扩展版可以同时返回相交状态和交点，分为四种情况：$0$ 代表不相交；$1$ 代表普通相交；$2$ 代表重叠（交于两个点）；$3$ 代表相交于端点。**需要注意**，部分运算可能会使用到直线求交点，此时务必保证变量类型为浮点数！

```c++
template<class T> tuple<int, Pt, Pt> segmentIntersection(Lt l1, Lt l2) {
    auto [s1, e1] = l1;
    auto [s2, e2] = l2;
    auto A = max(s1.x, e1.x), AA = min(s1.x, e1.x);
    auto B = max(s1.y, e1.y), BB = min(s1.y, e1.y);
    auto C = max(s2.x, e2.x), CC = min(s2.x, e2.x);
    auto D = max(s2.y, e2.y), DD = min(s2.y, e2.y);
    if (A < CC || C < AA || B < DD || D < BB) {
        return {0, {}, {}};
    }
    if (sign(cross(e1 - s1, e2 - s2)) == 0) {
        if (sign(cross(s2, e1, s1)) != 0) {
            return {0, {}, {}};
        }
        Pt p1(max(AA, CC), max(BB, DD));
        Pt p2(min(A, C), min(B, D));
        if (!pointOnSegment(p1, l1)) {
            swap(p1.y, p2.y);
        }
        if (p1 == p2) {
            return {3, p1, p2};
        } else {
            return {2, p1, p2};
        }
    }
    auto cp1 = cross(s2 - s1, e2 - s1);
    auto cp2 = cross(s2 - e1, e2 - e1);
    auto cp3 = cross(s1 - s2, e1 - s2);
    auto cp4 = cross(s1 - e2, e1 - e2);
    if (sign(cp1 * cp2) == 1 || sign(cp3 * cp4) == 1) {
        return {0, {}, {}};
    }
    // 使用下方函数时请使用浮点数
    Pd p = lineIntersection(l1, l2);
    if (sign(cp1) != 0 && sign(cp2) != 0 && sign(cp3) != 0 && sign(cp4) != 0) {
        return {1, p, p};
    } else {
        return {3, p, p};
    }
}
```

如果不需要求交点，那么使用快速排斥+跨立实验即可，其中重叠、相交于端点均视为相交。

```c++
template<class T> bool segmentIntersection(Lt l1, Lt l2) {
    auto [s1, e1] = l1;
    auto [s2, e2] = l2;
    auto A = max(s1.x, e1.x), AA = min(s1.x, e1.x);
    auto B = max(s1.y, e1.y), BB = min(s1.y, e1.y);
    auto C = max(s2.x, e2.x), CC = min(s2.x, e2.x);
    auto D = max(s2.y, e2.y), DD = min(s2.y, e2.y);
    return A >= CC && B >= DD && C >= AA && D >= BB &&
           sign(cross(s1, s2, e1) * cross(s1, e1, e2)) == 1 &&
           sign(cross(s2, s1, e2) * cross(s2, e2, e1)) == 1;
}
```

## 5.5 平面圆相关（浮点数处理）

### 5.5.1 点到圆的最近点

同时返回最近点与最近距离。**需要注意**，当点为圆心时，这样的点有无数个，此时我们视作输入错误，直接返回圆心。

```c++
pair<Pd, ld> pointToCircle(Pd p, Pd o, ld r) {
    Pd U = o, V = o;
    ld d = dis(p, o);
    if (sign(d) == 0) { // p 为圆心时返回圆心本身
        return {o, 0};
    }
    ld val1 = r * abs(o.x - p.x) / d;
    ld val2 = r * abs(o.y - p.y) / d * ((o.x - p.x) * (o.y - p.y) < 0 ? -1 : 1);
    U.x += val1, U.y += val2;
    V.x -= val1, V.y -= val2;
    if (dis(U, p) < dis(V, p)) {
        return {U, dis(U, p)};
    } else {
        return {V, dis(V, p)};
    }
}
```

### 5.5.2 根据圆心角获取圆上某点

将圆上最右侧的点以圆心为旋转中心，逆时针旋转 `rad` 度。

```c++
Point<ld> getPoint(Point<ld> p, ld r, ld rad) {
    return {p.x + cos(rad) * r, p.y + sin(rad) * r};
}
```

### 5.5.3 直线是否与圆相交及交点

$0$ 代表不相交；$1$ 代表相切；$2$ 代表相交。

```c++
tuple<int, Pd, Pd> lineCircleCross(Ld l, Pd o, ld r) {
    Pd P = project(o, l);
    ld d = dis(P, o), tmp = r * r - d * d;
    if (sign(tmp) == -1) {
        return {0, {}, {}};
    } else if (sign(tmp) == 0) {
        return {1, P, {}};
    }
    Pd vec = standardize(l.b - l.a) * sqrt(tmp);
    return {2, P + vec, P - vec};
}
```

### 5.5.4 线段是否与圆相交及交点

$0$ 代表不相交；$1$ 代表相切；$2$ 代表相交于一个点；$3$ 代表相交于两个点。

```c++
tuple<int, Pd, Pd> segmentCircleCross(Ld l, Pd o, ld r) {
    auto [type, U, V] = lineCircleCross(l, o, r);
    bool f1 = pointOnSegment(U, l), f2 = pointOnSegment(V, l);
    if (type == 1 && f1) {
        return {1, U, {}};
    } else if (type == 2 && f1 && f2) {
        return {3, U, V};
    } else if (type == 2 && f1) {
        return {2, U, {}};
    } else if (type == 2 && f2) {
        return {2, V, {}};
    } else {
        return {0, {}, {}};
    }
}
```

### 5.5.5 两圆是否相交及交点

$0$ 代表内含；$1$ 代表相离；$2$ 代表相切；$3$ 代表相交。

```c++
tuple<int, Pd, Pd> circleIntersection(Pd p1, ld r1, Pd p2, ld r2) {
    ld x1 = p1.x, x2 = p2.x, y1 = p1.y, y2 = p2.y, d = dis(p1, p2);
    if (sign(abs(r1 - r2) - d) == 1) {
        return {0, {}, {}};
    } else if (sign(r1 + r2 - d) == -1) {
        return {1, {}, {}};
    }
    ld a = r1 * (x1 - x2) * 2, b = r1 * (y1 - y2) * 2, c = r2 * r2 - r1 * r1 - d * d;
    ld p = a * a + b * b, q = -a * c * 2, r = c * c - b * b;
    ld cosa, sina, cosb, sinb;
    if (sign(d - (r1 + r2)) == 0 || sign(d - abs(r1 - r2)) == 0) {
        cosa = -q / p / 2;
        sina = sqrt(1 - cosa * cosa);
        Point<ld> p0 = {x1 + r1 * cosa, y1 + r1 * sina};
        if (sign(dis(p0, p2) - r2)) {
            p0.y = y1 - r1 * sina;
        }
        return {2, p0, p0};
    } else {
        ld delta = sqrt(q * q - p * r * 4);
        cosa = (delta - q) / p / 2;
        cosb = (-delta - q) / p / 2;
        sina = sqrt(1 - cosa * cosa);
        sinb = sqrt(1 - cosb * cosb);
        Pd ans1 = {x1 + r1 * cosa, y1 + r1 * sina};
        Pd ans2 = {x1 + r1 * cosb, y1 + r1 * sinb};
        if (sign(dis(ans1, p1) - r2)) ans1.y = y1 - r1 * sina;
        if (sign(dis(ans2, p2) - r2)) ans2.y = y1 - r1 * sinb;
        if (ans1 == ans2) ans1.y = y1 - r1 * sina;
        return {3, ans1, ans2};
    }
}
```

### 5.5.6 两圆相交面积

上述所言四种相交情况均可计算，之所以不使用三角形面积计算公式是因为在计算过程中会出现“负数”面积（扇形面积与三角形面积的符号关系会随圆的位置关系发生变化），故公式全部重新推导，这里采用的是扇形面积减去扇形内部的那个三角形的面积。

```c++
ld circleIntersectionArea(Pd p1, ld r1, Pd p2, ld r2) {
    ld x1 = p1.x, x2 = p2.x, y1 = p1.y, y2 = p2.y, d = dis(p1, p2);
    if (sign(abs(r1 - r2) - d) >= 0) {
        return PI * min(r1 * r1, r2 * r2);
    } else if (sign(r1 + r2 - d) == -1) {
        return 0;
    }
    ld theta1 = angle(r1, dis(p1, p2), r2);
    ld area1 = r1 * r1 * (theta1 - sin(theta1 * 2) / 2);
    ld theta2 = angle(r2, dis(p1, p2), r1);
    ld area2 = r2 * r2 * (theta2 - sin(theta2 * 2) / 2);
    return area1 + area2;
}
```

### 5.5.7 三点确定一圆

```c++
tuple<int, Pd, ld> getCircle(Pd A, Pd B, Pd C) {
    if (onLine(A, B, C)) { // 特判三点共线
        return {0, {}, 0};
    }
    Ld l1 = midSegment(Line{A, B});
    Ld l2 = midSegment(Line{A, C});
    Pd O = lineIntersection(l1, l2);
    return {1, O, dis(A, O)};
}
```

### 5.5.8 求解点到圆的切线数量与切点

```c++
pair<int, vector<Point<ld>>> tangent(Point<ld> p, Point<ld> A, ld r) {
    vector<Point<ld>> ans; // 储存切点
    Point<ld> u = A - p;
    ld d = sqrt(dot(u, u));
    if (d < r) {
        return {0, {}};
    } else if (sign(d - r) == 0) { // 点在圆上
        ans.push_back(u);
        return {1, ans};
    } else {
        ld ang = asin(r / d);
        ans.push_back(getPoint(A, r, -ang));
        ans.push_back(getPoint(A, r, ang));
        return {2, ans};
    }
}
```

### 5.5.9 求解两圆的内公、外公切线数量与切点

同时返回公切线数量以及每个圆的切点。

```c++
tuple<int, vector<Point<ld>>, vector<Point<ld>>> tangent(Point<ld> A, ld Ar, Point<ld> B, ld Br) {
    vector<Point<ld>> a, b; // 储存切点
    if (Ar < Br) {
        swap(Ar, Br);
        swap(A, B);
        swap(a, b);
    }
    int d = disEx(A, B), dif = Ar - Br, sum = Ar + Br;
    if (d < dif * dif) { // 内含，无
        return {0, {}, {}};
    }
    ld base = atan2(B.y - A.y, B.x - A.x);
    if (d == 0 && Ar == Br) { // 完全重合，无数条外公切线
        return {-1, {}, {}};
    }
    if (d == dif * dif) { // 内切，1条外公切线
        a.push_back(getPoint(A, Ar, base));
        b.push_back(getPoint(B, Br, base));
        return {1, a, b};
    }
    ld ang = acos(dif / sqrt(d));
    a.push_back(getPoint(A, Ar, base + ang)); // 保底2条外公切线
    a.push_back(getPoint(A, Ar, base - ang));
    b.push_back(getPoint(B, Br, base + ang));
    b.push_back(getPoint(B, Br, base - ang));
    if (d == sum * sum) { // 外切，多1条内公切线
        a.push_back(getPoint(A, Ar, base));
        b.push_back(getPoint(B, Br, base + PI));
    } else if (d > sum * sum) { // 相离，多2条内公切线
        ang = acos(sum / sqrt(d));
        a.push_back(getPoint(A, Ar, base + ang));
        a.push_back(getPoint(A, Ar, base - ang));
        b.push_back(getPoint(B, Br, base + ang + PI));
        b.push_back(getPoint(B, Br, base - ang + PI));
    }
    return {a.size(), a, b};
}
```

## 5.6 平面三角形相关（浮点数处理）

### 5.6.1 三角形面积

```c++
ld area(Point<ld> a, Point<ld> b, Point<ld> c) {
    return abs(cross(b, c, a)) / 2;
}
```

### 5.6.2 三角形外心

三角形外接圆的圆心，即三角形三边垂直平分线的交点。

```c++
template<class T> Pt center1(Pt p1, Pt p2, Pt p3) { // 外心
    return lineIntersection(midSegment({p1, p2}), midSegment({p2, p3}));
}
```

### 5.6.3 三角形内心

三角形内切圆的圆心，也是三角形三个内角的角平分线的交点。其到三角形三边的距离相等。

```c++
Pd center2(Pd p1, Pd p2, Pd p3) { // 内心
    #define atan2(p) atan2(p.y, p.x) // 注意先后顺序
    Line<ld> U = {p1, {}}, V = {p2, {}};
    ld m, n, alpha;
    m = atan2((p2 - p1));
    n = atan2((p3 - p1));
    alpha = (m + n) / 2;
    U.b = {p1.x + cos(alpha), p1.y + sin(alpha)};
    m = atan2((p1 - p2));
    n = atan2((p3 - p2));
    alpha = (m + n) / 2;
    V.b = {p2.x + cos(alpha), p2.y + sin(alpha)};
    return lineIntersection(U, V);
}
```

### 5.6.4 三角形垂心

三角形的三条高线所在直线的交点。锐角三角形的垂心在三角形内；直角三角形的垂心在直角顶点上；钝角三角形的垂心在三角形外。

```c++
Pd center3(Pd p1, Pd p2, Pd p3) { // 垂心
    Ld U = {p1, p1 + rotate(p2, p3)}; // 垂线
    Ld V = {p2, p2 + rotate(p1, p3)};
    return lineIntersection(U, V);
}
```

## 5.7 平面直线方程转换

### 5.7.1 浮点数计算直线的斜率

一般很少使用到这个函数，因为斜率的取值不可控（例如接近平行于 $x,y$ 轴时）。**需要注意**，当直线平行于 $y$ 轴时斜率为 `inf` 。

```c++
template <class T> ld slope(Pt p1, Pt p2) { // 斜率，注意 inf 的情况
    return (p1.y - p2.y) / (p1.x - p2.x);
}
template <class T> ld slope(Lt l) {
    return slope(l.a, l.b);
}
```

### 5.7.2 分数精确计算直线的斜率

调用分数四则运算精确计算斜率，返回最简分数，只适用于整数计算。

```c++
template<class T> Frac<T> slopeEx(Pt p1, Pt p2) {
    Frac<T> U = p1.y - p2.y;
    Frac<T> V = p1.x - p2.x;
    return U / V; // 调用分数精确计算
}
```

### 5.7.3 两点式转一般式

返回由三个整数构成的方程，在输入较大时可能找不到较小的满足题意的一组整数解。可以处理平行于 $x,y$ 轴、两点共点的情况。

```c++
template<class T> tuple<T, T, T> getfun(Lt p) {
    T A = p.a.y - p.b.y, B = p.b.x - p.a.x, C = p.a.x * A + p.a.y * B;
    if (A < 0) { // 符号调整
        A = -A, B = -B, C = -C;
    } else if (A == 0) {
        if (B < 0) {
            B = -B, C = -C;
        } else if (B == 0 && C < 0) {
            C = -C;
        }
    }
    if (A == 0) { // 数值计算
        if (B == 0) {
            C = 0; // 共点特判
        } else {
            T g = fgcd(abs(B), abs(C));
            B /= g, C /= g;
        }
    } else if (B == 0) {
        T g = fgcd(abs(A), abs(C));
        A /= g, C /= g;
    } else {
        T g = fgcd(fgcd(abs(A), abs(B)), abs(C));
        A /= g, B /= g, C /= g;
    }
    return tuple{A, B, C}; // Ax + By = C
}
```

### 5.7.4 一般式转两点式

由于整数点可能很大或者不存在，故直接采用浮点数；如果与 $x,y$ 轴有交点则取交点。可以处理平行于 $x,y$ 轴的情况。

```c++
Line<ld> getfun(int A, int B, int C) { // Ax + By = C
    ld x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    if (A && B) { // 正常
        if (C) {
            x1 = 0, y1 = 1. * C / B;
            y2 = 0, x2 = 1. * C / A;
        } else { // 过原点
            x1 = 1, y1 = 1. * -A / B;
            x2 = 0, y2 = 0;
        }
    } else if (A && !B) { // 垂直
        if (C) {
            y1 = 0, x1 = 1. * C / A;
            y2 = 1, x2 = x1;
        } else {
            x1 = 0, y1 = 1;
            x2 = 0, y2 = 0;
        }
    } else if (!A && B) { // 水平
        if (C) {
            x1 = 0, y1 = 1. * C / B;
            x2 = 1, y2 = y1;
        } else {
            x1 = 1, y1 = 0;
            x2 = 0, y2 = 0;
        }
    } else { // 不合法，请特判
        assert(false);
    }
    return {{x1, y1}, {x2, y2}};
}
```

### 5.7.5 抛物线与 x 轴是否相交及交点

$0$ 代表没有交点；$1$ 代表相切；$2$ 代表有两个交点。

```c++
tuple<int, ld, ld> getAns(ld a, ld b, ld c) {
    ld delta = b * b - a * c * 4;
    if (delta < 0.) {
        return {0, 0, 0};
    }
    delta = sqrt(delta);
    ld ans1 = -(delta + b) / 2 / a;
    ld ans2 = (delta - b) / 2 / a;
    if (ans1 > ans2) {
        swap(ans1, ans2);
    }
    if (sign(delta) == 0) {
        return {1, ans2, 0};
    }
    return {2, ans1, ans2};
}
```

## 5.8 平面多边形

### 5.8.1 两向量构成的平面四边形有向面积

```c++
template<class T> T areaEx(Point<T> p1, Point<T> p2, Point<T> p3) {
    return cross(b, c, a);
}
```

### 5.8.2 判断四个点能否组成矩形/正方形

可以处理浮点数、共点的情况。返回分为三种情况：$2$ 代表构成正方形；$1$ 代表构成矩形；$0$ 代表其他情况。

```c++
template<class T> int isSquare(vector<Pt> x) {
    sort(x.begin(), x.end());
    if (equal(dis(x[0], x[1]), dis(x[2], x[3])) && sign(dis(x[0], x[1])) &&
        equal(dis(x[0], x[2]), dis(x[1], x[3])) && sign(dis(x[0], x[2])) &&
        lineParallel(Lt{x[0], x[1]}, Lt{x[2], x[3]}) &&
        lineParallel(Lt{x[0], x[2]}, Lt{x[1], x[3]}) &&
        lineVertical(Lt{x[0], x[1]}, Lt{x[0], x[2]})) {
        return equal(dis(x[0], x[1]), dis(x[0], x[2])) ? 2 : 1;
    }
    return 0;
}
```

### 5.8.3 点是否在任意多边形内

射线法判定，$t$ 为穿越次数，当其为奇数时即代表点在多边形内部；返回 $2$ 代表点在多边形边界上。

```c++
template<class T> int pointInPolygon(Point<T> a, vector<Point<T>> p) {
    int n = p.size();
    for (int i = 0; i < n; i++) {
        if (pointOnSegment(a, Line{p[i], p[(i + 1) % n]})) {
            return 2;
        }
    }
    int t = 0;
    for (int i = 0; i < n; i++) {
        auto u = p[i], v = p[(i + 1) % n];
        if (u.x < a.x && v.x >= a.x && pointOnLineLeft(a, Line{v, u})) {
            t ^= 1;
        }
        if (u.x >= a.x && v.x < a.x && pointOnLineLeft(a, Line{u, v})) {
            t ^= 1;
        }
    }
    return t == 1;
}
```

### 5.8.4 线段是否在任意多边形内部

```c++
template<class T>
bool segmentInPolygon(Line<T> l, vector<Point<T>> p) {
// 线段与多边形边界不相交且两端点都在多边形内部
#define L(x, y) pointOnLineLeft(x, y)
    int n = p.size();
    if (!pointInPolygon(l.a, p)) return false;
    if (!pointInPolygon(l.b, p)) return false;
    for (int i = 0; i < n; i++) {
        auto u = p[i];
        auto v = p[(i + 1) % n];
        auto w = p[(i + 2) % n];
        auto [t, p1, p2] = segmentIntersection(l, Line(u, v));
        if (t == 1) return false;
        if (t == 0) continue;
        if (t == 2) {
            if (pointOnSegment(v, l) && v != l.a && v != l.b) {
                if (cross(v - u, w - v) > 0) {
                    return false;
                }
            }
        } else {
            if (p1 != u && p1 != v) {
                if (L(l.a, Line(v, u)) || L(l.b, Line(v, u))) {
                    return false;
                }
            } else if (p1 == v) {
                if (l.a == v) {
                    if (L(u, l)) {
                        if (L(w, l) && L(w, Line(u, v))) {
                            return false;
                        }
                    } else {
                        if (L(w, l) || L(w, Line(u, v))) {
                            return false;
                        }
                    }
                } else if (l.b == v) {
                    if (L(u, Line(l.b, l.a))) {
                        if (L(w, Line(l.b, l.a)) && L(w, Line(u, v))) {
                            return false;
                        }
                    } else {
                        if (L(w, Line(l.b, l.a)) || L(w, Line(u, v))) {
                            return false;
                        }
                    }
                } else {
                    if (L(u, l)) {
                        if (L(w, Line(l.b, l.a)) || L(w, Line(u, v))) {
                            return false;
                        }
                    } else {
                        if (L(w, l) || L(w, Line(u, v))) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}
```

### 5.8.5 任意多边形的面积

```c++
template<class T> ld area(vector<Point<T>> P) {
    int n = P.size();
    ld ans = 0;
    for (int i = 0; i < n; i++) {
        ans += cross(P[i], P[(i + 1) % n]);
    }
    return ans / 2.0;
}
```

### 5.8.6 皮克定理

绘制在方格纸上的多边形面积公式可以表示为 $S=n+\dfrac{s}{2}-1$ ，其中 $n$ 表示多边形内部的点数、$s$ 表示多边形边界上的点数。一条线段上的点数为 $\gcd(|x_1-x_2|,|y_1-y_2|)+1$。

### 5.8.7 任意多边形上/内的网格点个数（仅能处理整数）

皮克定理用。

```c++
int onPolygonGrid(vector<Point<int>> p) { // 多边形上
    int n = p.size(), ans = 0;
    for (int i = 0; i < n; i++) {
        auto a = p[i], b = p[(i + 1) % n];
        ans += gcd(abs(a.x - b.x), abs(a.y - b.y));
    }
    return ans;
}
int inPolygonGrid(vector<Point<int>> p) { // 多边形内
    int n = p.size(), ans = 0;
    for (int i = 0; i < n; i++) {
        auto a = p[i], b = p[(i + 1) % n], c = p[(i + 2) % n];
        ans += b.y * (a.x - c.x);
    }
    ans = abs(ans);
    return (ans - onPolygonGrid(p)) / 2 + 1;
}
```

## 5.9 二维凸包

### 5.9.1 获取二维静态凸包（Andrew算法）

`flag` 用于判定凸包边上的点、重复的顶点是否要加入到凸包中，为 $0$ 时代表加入凸包（不严格）；为 $1$ 时不加入凸包（严格）。时间复杂度为 $\mathcal O(N\log N)$ 。

```c++
template<class T> vector<Point<T>> staticConvexHull(vector<Point<T>> A, int flag = 1) {
    int n = A.size();
    if (n <= 2) { // 特判
        return A;
    }
    vector<Point<T>> ans(n * 2);
    sort(A.begin(), A.end());
    int now = -1;
    for (int i = 0; i < n; i++) { // 维护下凸包
        while (now > 0 && cross(A[i], ans[now], ans[now - 1]) <= 0) {
            now--;
        }
        ans[++now] = A[i];
    }
    int pre = now;
    for (int i = n - 2; i >= 0; i--) { // 维护上凸包
        while (now > pre && cross(A[i], ans[now], ans[now - 1]) <= 0) {
            now--;
        }
        ans[++now] = A[i];
    }
    ans.resize(now);
    return ans;
}
```

### 5.9.2 二维动态凸包

固定为 `int` 型，需要重新书写 `Line` 函数，`cmp` 用于判定边界情况。可以处理如下两个要求：

- 动态插入点 $(x,y)$ 到当前凸包中；
- 判断点 $(x,y)$ 是否在凸包上或是在内部（包括边界）。

```c++
template<class T> bool turnRight(Pt a, Pt b) {
    return cross(a, b) < 0 || (cross(a, b) == 0 && dot(a, b) < 0);
}
struct Line {
    static int cmp;
    mutable Point<int> a, b;
    friend bool operator<(Line x, Line y) {
        return cmp ? x.a < y.a : turnRight(x.b, y.b);
    }
    friend auto &operator<<(ostream &os, Line l) {
        return os << "<" << l.a << ", " << l.b << ">";
    }
};

int Line::cmp = 1;
struct UpperConvexHull : set<Line> {
    bool contains(const Point<int> &p) const {
        auto it = lower_bound({p, 0});
        if (it != end() && it->a == p) return true;
        if (it != begin() && it != end() && cross(prev(it)->b, p - prev(it)->a) <= 0) {
            return true;
        }
        return false;
    }
    void add(const Point<int> &p) {
        if (contains(p)) return;
        auto it = lower_bound({p, 0});
        for (; it != end(); it = erase(it)) {
            if (turnRight(it->a - p, it->b)) {
                break;
            }
        }
        for (; it != begin() && prev(it) != begin(); erase(prev(it))) {
            if (turnRight(prev(prev(it))->b, p - prev(prev(it))->a)) {
                break;
            }
        }
        if (it != begin()) {
            prev(it)->b = p - prev(it)->a;
        }
        if (it == end()) {
            insert({p, {0, -1}});
        } else {
            insert({p, it->a - p});
        }
    }
};
struct ConvexHull {
    UpperConvexHull up, low;
    bool empty() const {
        return up.empty();
    }
    bool contains(const Point<int> &p) const {
        Line::cmp = 1;
        return up.contains(p) && low.contains(-p);
    }
    void add(const Point<int> &p) {
        Line::cmp = 1;
        up.add(p);
        low.add(-p);
    }
    bool isIntersect(int A, int B, int C) const {
        Line::cmp = 0;
        if (empty()) return false;
        Point<int> k = {-B, A};
        if (k.x < 0) k = -k;
        if (k.x == 0 && k.y < 0) k.y = -k.y;
        Point<int> P = up.upper_bound({{0, 0}, k})->a;
        Point<int> Q = -low.upper_bound({{0, 0}, k})->a;
        return sign(A * P.x + B * P.y - C) * sign(A * Q.x + B * Q.y - C) > 0;
    }
    friend ostream &operator<<(ostream &out, const ConvexHull &ch) {
        for (const auto &line : ch.up) out << "(" << line.a.x << "," << line.a.y << ")";
        cout << "/";
        for (const auto &line : ch.low) out << "(" << -line.a.x << "," << -line.a.y << ")";
        return out;
    }
};
```

### 5.9.3 点与凸包的位置关系

$0$ 代表点在凸包外面；$1$ 代表在凸壳上；$2$ 代表在凸包内部。

```c++
template<class T> int contains(Point<T> p, vector<Point<T>> A) {
    int n = A.size();
    bool in = false;
    for (int i = 0; i < n; i++) {
        Point<T> a = A[i] - p, b = A[(i + 1) % n] - p;
        if (a.y > b.y) {
            swap(a, b);
        }
        if (a.y <= 0 && 0 < b.y && cross(a, b) < 0) {
            in = !in;
        }
        if (cross(a, b) == 0 && dot(a, b) <= 0) {
            return 1;
        }
    }
    return in ? 2 : 0;
}
```

### 5.9.4 闵可夫斯基和

计算两个凸包合成的大凸包。

```c++
template<class T> vector<Point<T>> mincowski(vector<Point<T>> P1, vector<Point<T>> P2) {
    int n = P1.size(), m = P2.size();
    vector<Point<T>> V1(n), V2(m);
    for (int i = 0; i < n; i++) {
        V1[i] = P1[(i + 1) % n] - P1[i];
    }
    for (int i = 0; i < m; i++) {
        V2[i] = P2[(i + 1) % m] - P2[i];
    }
    vector<Point<T>> ans = {P1.front() + P2.front()};
    int t = 0, i = 0, j = 0;
    while (i < n && j < m) {
        Point<T> val = sign(cross(V1[i], V2[j])) > 0 ? V1[i++] : V2[j++];
        ans.push_back(ans.back() + val);
    }
    while (i < n) ans.push_back(ans.back() + V1[i++]);
    while (j < m) ans.push_back(ans.back() + V2[j++]);
    return ans;
}
```

### 5.9.5 半平面交

计算多条直线左边平面部分的交集。

```c++
template<class T> vector<Point<T>> halfcut(vector<Line<T>> lines) {
    sort(lines.begin(), lines.end(), [&](auto l1, auto l2) {
        auto d1 = l1.b - l1.a;
        auto d2 = l2.b - l2.a;
        if (sign(d1) != sign(d2)) {
            return sign(d1) == 1;
        }
        return cross(d1, d2) > 0;
    });
    deque<Line<T>> ls;
    deque<Point<T>> ps;
    for (auto l : lines) {
        if (ls.empty()) {
            ls.push_back(l);
            continue;
        }
        while (!ps.empty() && !pointOnLineLeft(ps.back(), l)) {
            ps.pop_back();
            ls.pop_back();
        }
        while (!ps.empty() && !pointOnLineLeft(ps[0], l)) {
            ps.pop_front();
            ls.pop_front();
        }
        if (cross(l.b - l.a, ls.back().b - ls.back().a) == 0) {
            if (dot(l.b - l.a, ls.back().b - ls.back().a) > 0) {
                if (!pointOnLineLeft(ls.back().a, l)) {
                    assert(ls.size() == 1);
                    ls[0] = l;
                }
                continue;
            }
            return {};
        }
        ps.push_back(lineIntersection(ls.back(), l));
        ls.push_back(l);
    }
    while (!ps.empty() && !pointOnLineLeft(ps.back(), ls[0])) {
        ps.pop_back();
        ls.pop_back();
    }
    if (ls.size() <= 2) {
        return {};
    }
    ps.push_back(lineIntersection(ls[0], ls.back()));
    return vector(ps.begin(), ps.end());
}
```

## 5.10 三维几何必要初始化

### 5.10.1 点线面封装

```c++
struct Point3 {
    ld x, y, z;
    Point3(ld x_ = 0, ld y_ = 0, ld z_ = 0) : x(x_), y(y_), z(z_) {}
    Point3 &operator+=(Point3 p) & {
        return x += p.x, y += p.y, z += p.z, *this;
    }
    Point3 &operator-=(Point3 p) & {
        return x -= p.x, y -= p.y, z -= p.z, *this;
    }
    Point3 &operator*=(Point3 p) & {
        return x *= p.x, y *= p.y, z *= p.z, *this;
    }
    Point3 &operator*=(ld t) & {
        return x *= t, y *= t, z *= t, *this;
    }
    Point3 &operator/=(ld t) & {
        return x /= t, y /= t, z /= t, *this;
    }
    friend Point3 operator+(Point3 a, Point3 b) { return a += b; }
    friend Point3 operator-(Point3 a, Point3 b) { return a -= b; }
    friend Point3 operator*(Point3 a, Point3 b) { return a *= b; }
    friend Point3 operator*(Point3 a, ld b) { return a *= b; }
    friend Point3 operator*(ld a, Point3 b) { return b *= a; }
    friend Point3 operator/(Point3 a, ld b) { return a /= b; }
    friend auto &operator>>(istream &is, Point3 &p) {
        return is >> p.x >> p.y >> p.z;
    }
    friend auto &operator<<(ostream &os, Point3 p) {
        return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
    }
};
struct Line3 {
    Point3 a, b;
};
struct Plane {
    Point3 u, v, w;
};
```

### 5.10.2 其他函数

```c++
ld len(P3 p) { // 原点到当前点的距离计算
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
P3 crossEx(P3 a, P3 b) { // 叉乘
    P3 ans;
    ans.x = a.y * b.z - a.z * b.y;
    ans.y = a.z * b.x - a.x * b.z;
    ans.z = a.x * b.y - a.y * b.x;
    return ans;
}
ld cross(P3 a, P3 b) {
    return len(crossEx(a, b));
}
ld dot(P3 a, P3 b) { // 点乘
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
P3 getVec(Plane s) { // 获取平面法向量
    return crossEx(s.u - s.v, s.v - s.w);
}
ld dis(P3 a, P3 b) { // 三维欧几里得距离公式
    ld val = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    return sqrt(val);
}
P3 standardize(P3 vec) { // 将三维向量转换为单位向量
    return vec / len(vec);
}
```

## 5.11 三维点线面相关

### 5.11.1 空间三点是否共线

其中第二个函数是专门用来判断给定的三个点能否构成平面的，因为不共线的三点才能构成平面。

```c++
bool onLine(P3 p1, P3 p2, P3 p3) { // 三点是否共线
    return sign(cross(p1 - p2, p3 - p2)) == 0;
}
bool onLine(Plane s) {
    return onLine(s.u, s.v, s.w);
}
```

### 5.11.2 四点是否共面

```c++
bool onPlane(P3 p1, P3 p2, P3 p3, P3 p4) { // 四点是否共面
    ld val = dot(getVec({p1, p2, p3}), p4 - p1);
    return sign(val) == 0;
}
```

### 5.11.3 空间点是否在线段上

```c++
bool pointOnSegment(P3 p, L3 l) {
    return sign(cross(p - l.a, p - l.b)) == 0 && min(l.a.x, l.b.x) <= p.x &&
           p.x <= max(l.a.x, l.b.x) && min(l.a.y, l.b.y) <= p.y && p.y <= max(l.a.y, l.b.y) &&
           min(l.a.z, l.b.z) <= p.z && p.z <= max(l.a.z, l.b.z);
}
bool pointOnSegmentEx(P3 p, L3 l) { // pointOnSegment去除端点版
    return sign(cross(p - l.a, p - l.b)) == 0 && min(l.a.x, l.b.x) < p.x &&
           p.x < max(l.a.x, l.b.x) && min(l.a.y, l.b.y) < p.y && p.y < max(l.a.y, l.b.y) &&
           min(l.a.z, l.b.z) < p.z && p.z < max(l.a.z, l.b.z);
}
```

### 5.11.4 空间两点是否在线段同侧

当给定的两点与线段不共面、点在线段上时返回 $false$ 。

```c++
bool pointOnSegmentSide(P3 p1, P3 p2, L3 l) {
    if (!onPlane(p1, p2, l.a, l.b)) { // 特判不共面
        return 0;
    }
    ld val = dot(crossEx(l.a - l.b, p1 - l.b), crossEx(l.a - l.b, p2 - l.b));
    return sign(val) == 1;
}
```

### 5.11.5 两点是否在平面同侧

点在平面上时返回 $false$ 。

```c++
bool pointOnPlaneSide(P3 p1, P3 p2, Plane s) {
    ld val = dot(getVec(s), p1 - s.u) * dot(getVec(s), p2 - s.u);
    return sign(val) == 1;
}
```

### 5.11.6 空间两直线是否平行/垂直

```c++
bool lineParallel(L3 l1, L3 l2) {
    return sign(cross(l1.a - l1.b, l2.a - l2.b)) == 0;
}
bool lineVertical(L3 l1, L3 l2) {
    return sign(dot(l1.a - l1.b, l2.a - l2.b)) == 0;
}
```

### 5.11.7 两平面是否平行/垂直

```c++
bool planeParallel(Plane s1, Plane s2) {
    ld val = cross(getVec(s1), getVec(s2));
    return sign(val) == 0;
}
bool planeVertical(Plane s1, Plane s2) {
    ld val = dot(getVec(s1), getVec(s2));
    return sign(val) == 0;
}
```

### 5.11.8 空间两直线是否是同一条

```c++
bool same(L3 l1, L3 l2) {
    return lineParallel(l1, l2) && lineParallel({l1.a, l2.b}, {l1.b, l2.a});
}
```

### 5.11.9 两平面是否是同一个

```c++
bool same(Plane s1, Plane s2) {
    return onPlane(s1.u, s2.u, s2.v, s2.w) && onPlane(s1.v, s2.u, s2.v, s2.w) &&
           onPlane(s1.w, s2.u, s2.v, s2.w);
}
```

### 5.11.10 直线是否与平面平行

```c++
bool linePlaneParallel(L3 l, Plane s) {
    ld val = dot(l.a - l.b, getVec(s));
    return sign(val) == 0;
}
```

### 5.11.11 空间两线段是否相交

```c++
bool segmentIntersection(L3 l1, L3 l2) { // 重叠、相交于端点均视为相交
    if (!onPlane(l1.a, l1.b, l2.a, l2.b)) { // 特判不共面
        return 0;
    }
    if (!onLine(l1.a, l1.b, l2.a) || !onLine(l1.a, l1.b, l2.b)) {
        return !pointOnSegmentSide(l1.a, l1.b, l2) && !pointOnSegmentSide(l2.a, l2.b, l1);
    }
    return pointOnSegment(l1.a, l2) || pointOnSegment(l1.b, l2) || pointOnSegment(l2.a, l1) ||
           pointOnSegment(l2.b, l2);
}
bool segmentIntersection1(L3 l1, L3 l2) { // 重叠、相交于端点不视为相交
    return onPlane(l1.a, l1.b, l2.a, l2.b) && !pointOnSegmentSide(l1.a, l1.b, l2) &&
           !pointOnSegmentSide(l2.a, l2.b, l1);
}
```

### 5.11.12 空间两直线是否相交及交点

当两直线不共面、两直线平行时返回 $false$ 。

```c++
pair<bool, P3> lineIntersection(L3 l1, L3 l2) {
    if (!onPlane(l1.a, l1.b, l2.a, l2.b) || lineParallel(l1, l2)) {
        return {0, {}};
    }
    auto [s1, e1] = l1;
    auto [s2, e2] = l2;
    ld val = 0;
    if (!onPlane(l1.a, l1.b, {0, 0, 0}, {0, 0, 1})) {
        val = ((s1.x - s2.x) * (s2.y - e2.y) - (s1.y - s2.y) * (s2.x - e2.x)) /
              ((s1.x - e1.x) * (s2.y - e2.y) - (s1.y - e1.y) * (s2.x - e2.x));
    } else if (!onPlane(l1.a, l1.b, {0, 0, 0}, {0, 1, 0})) {
        val = ((s1.x - s2.x) * (s2.z - e2.z) - (s1.z - s2.z) * (s2.x - e2.x)) /
              ((s1.x - e1.x) * (s2.z - e2.z) - (s1.z - e1.z) * (s2.x - e2.x));
    } else {
        val = ((s1.y - s2.y) * (s2.z - e2.z) - (s1.z - s2.z) * (s2.y - e2.y)) /
              ((s1.y - e1.y) * (s2.z - e2.z) - (s1.z - e1.z) * (s2.y - e2.y));
    }
    return {1, s1 + (e1 - s1) * val};
}
```

### 5.11.13 直线与平面是否相交及交点

当直线与平面平行、给定的点构不成平面时返回 $false$ 。

```c++
pair<bool, P3> linePlaneCross(L3 l, Plane s) {
    if (linePlaneParallel(l, s)) {
        return {0, {}};
    }
    P3 vec = getVec(s);
    P3 U = vec * (s.u - l.a), V = vec * (l.b - l.a);
    ld val = (U.x + U.y + U.z) / (V.x + V.y + V.z);
    return {1, l.a + (l.b - l.a) * val};
}
```

### 5.11.14 两平面是否相交及交线

当两平面平行、两平面为同一个时返回 $false$ 。

```c++
pair<bool, L3> planeIntersection(Plane s1, Plane s2) {
    if (planeParallel(s1, s2) || same(s1, s2)) {
        return {0, {}};
    }
    P3 U = linePlaneParallel({s2.u, s2.v}, s1) ? linePlaneCross({s2.v, s2.w}, s1).second
                                               : linePlaneCross({s2.u, s2.v}, s1).second;
    P3 V = linePlaneParallel({s2.w, s2.u}, s1) ? linePlaneCross({s2.v, s2.w}, s1).second
                                               : linePlaneCross({s2.w, s2.u}, s1).second;
    return {1, {U, V}};
}
```

### 5.11.15 点到直线的最近点与最近距离

```c++
pair<ld, P3> pointToLine(P3 p, L3 l) {
    ld val = cross(p - l.a, l.a - l.b) / dis(l.a, l.b); // 面积除以底边长
    ld val1 = dot(p - l.a, l.a - l.b) / dis(l.a, l.b);
    return {val, l.a + val1 * standardize(l.a - l.b)};
}
```

### 5.11.16 点到平面的最近点与最近距离

```c++
pair<ld, P3> pointToPlane(P3 p, Plane s) {
    P3 vec = getVec(s);
    ld val = dot(vec, p - s.u);
    val = abs(val) / len(vec); // 面积除以底边长
    return {val, p - val * standardize(vec)};
}
```

### 5.11.17 空间两直线的最近距离与最近点对

```c++
tuple<ld, P3, P3> lineToLine(L3 l1, L3 l2) {
    P3 vec = crossEx(l1.a - l1.b, l2.a - l2.b); // 计算同时垂直于两直线的向量
    ld val = abs(dot(l1.a - l2.a, vec)) / len(vec);
    P3 U = l1.b - l1.a, V = l2.b - l2.a;
    vec = crossEx(U, V);
    ld p = dot(vec, vec);
    ld t1 = dot(crossEx(l2.a - l1.a, V), vec) / p;
    ld t2 = dot(crossEx(l2.a - l1.a, U), vec) / p;
    return {val, l1.a + (l1.b - l1.a) * t1, l2.a + (l2.b - l2.a) * t2};
}
```

## 5.12 三维角度与弧度

### 5.12.1 空间两直线夹角的 cos 值

任意位置的空间两直线。

```c++
ld lineCos(L3 l1, L3 l2) {
    return dot(l1.a - l1.b, l2.a - l2.b) / len(l1.a - l1.b) / len(l2.a - l2.b);
}
```

### 5.12.2 空间两平面夹角的 cos 值

```c++
ld planeCos(Plane s1, Plane s2) {
    P3 U = getVec(s1), V = getVec(s2);
    return dot(U, V) / len(U) / len(V);
}
```

### 5.12.3 直线与平面夹角的 sin 值

```c++
ld linePlaneSin(L3 l, Plane s) {
    P3 vec = getVec(s);
    return dot(l.a - l.b, vec) / len(l.a - l.b) / len(vec);
}
```

## 5.13 空间多边形

### 5.13.1 正N棱锥体积公式

棱锥通用体积公式 $V=\dfrac{1}{3}Sh$ ，当其恰好是棱长为 $l$ 的正 $n$ 棱锥时，有公式 $\displaystyle V=\frac{l^3\cdot n}{12\tan \frac{\pi}{n}}\cdot\sqrt{1-\frac{1}{4\cdot \sin^2\frac{\pi}{n}}}$。

```c++
ld V(ld l, int n) { // 正n棱锥体积公式
    return l * l * l * n / (12 * tan(PI / n)) * sqrt(1 - 1 / (4 * sin(PI / n) * sin(PI / n)));
}
```

### 5.13.2 四面体体积

```c++
ld V(P3 a, P3 b, P3 c, P3 d) {
    return abs(dot(d - a, crossEx(b - a, c - a))) / 6;
}
```

### 5.13.3 点是否在空间三角形上

点位于边界上时返回 $false$ 。

```c++
bool pointOnTriangle(P3 p, P3 p1, P3 p2, P3 p3) {
    return pointOnSegmentSide(p, p1, {p2, p3}) && pointOnSegmentSide(p, p2, {p1, p3}) &&
           pointOnSegmentSide(p, p3, {p1, p2});
}
```

### 5.13.4 线段是否与空间三角形相交及交点

只有交点在空间三角形内部时才视作相交。

```c++
pair<bool, P3> segmentOnTriangle(P3 l, P3 r, P3 p1, P3 p2, P3 p3) {
    P3 x = crossEx(p2 - p1, p3 - p1);
    if (sign(dot(x, r - l)) == 0) {
        return {0, {}};
    }
    ld t = dot(x, p1 - l) / dot(x, r - l);
    if (t < 0 || t - 1 > 0) { // 不在线段上
        return {0, {}};
    }
    bool type = pointOnTriangle(l + (r - l) * t, p1, p2, p3);
    if (type) {
        return {1, l + (r - l) * t};
    } else {
        return {0, {}};
    }
}
```

### 5.13.5 空间三角形是否相交

相交线段在空间三角形内部时才视作相交。

```c++
bool triangleIntersection(vector<P3> a, vector<P3> b) {
    for (int i = 0; i < 3; i++) {
        if (segmentOnTriangle(b[i], b[(i + 1) % 3], a[0], a[1], a[2]).first) {
            return 1;
        }
        if (segmentOnTriangle(a[i], a[(i + 1) % 3], b[0], b[1], b[2]).first) {
            return 1;
        }
    }
    return 0;
}
```

## 5.14 常用结论

### 5.14.1 平面几何结论归档

- `hypot` 函数可以直接计算直角三角形的斜边长；
- **边心距**是指正多边形的外接圆圆心到正多边形某一边的距离，边长为 $s$ 的正 $n$ 角形的边心距公式为 $\displaystyle a=\frac{t}{2\cdot\tan \frac{\pi}{n}}$ ，外接圆半径为 $R$ 的正 $n$ 角形的边心距公式为 $a=R\cdot \cos \dfrac{\pi}{n}$ ；
- **三角形外接圆半径**为 $\dfrac{a}{2\sin A}=\dfrac{abc}{4S}$ ，其中 $S$ 为三角形面积，内切圆半径为 $\dfrac{2S}{a+b+c}$；
- 由小正三角形拼成的大正三角形，耗费的小三角形数量即为构成一条边的小三角形数量的平方。如下图，总数量即为 $4^2$ [See](https://codeforces.com/problemset/problem/559/A)。
  
  <img src="https://s2.loli.net/2023/08/17/p7kRACD4cTf3YxK.png" alt="91044c3ef9c959aae5be2e7d53c13dd0.png" style="zoom:30%;" />
- 正 $n$ 边形圆心角为 $\dfrac{360^{\circ}}{n}$ ，圆周角为 $\dfrac{180^{\circ}}{n}$ 。定义正 $n$ 边形上的三个顶点 $A,B$ 和 $C$（可以不相邻），使得 $\angle ABC=\theta$ ，当 $n\le 360$ 时，$\theta$ 可以取 $1^{\circ}$ 到 $179^{\circ}$ 间的任何一个整数 [See](https://codeforces.com/problemset/problem/1096/C)。
- 某一点 $B$ 到直线 $AC$ 的距离公式为 $\dfrac{|\vec{BA}\times \vec{BC}|}{|AC|}$ ，等价于 $\dfrac{|aX+bY+c|}{\sqrt{a^2+b^2}}$。
- `atan(y / x)` 函数仅用于计算第一、四象限的值，而 `atan2(y, x)` 则允许计算所有四个象限的正反切，在使用这个函数时，需要尽量保证 $x$ 和 $y$ 的类型为整数型，如果使用浮点数，实测会慢十倍。
- 在平面上有奇数个点 $A_0,A_1,\dots,A_n$ 以及一个点 $X_0$ ，构造 $X_1$ 使得 $X_0,X_1$ 关于 $A_0$ 对称、构造 $X_2$ 使得 $X_1,X_2$ 关于 $A_1$ 对称、……、构造 $X_j$ 使得 $X_{j-1},X_j$ 关于 $A_{(j-1)\mod n}$ 对称。那么周期为 $2n$ ，即 $A_0$ 与 $A_{2n}$ 共点、$A_1$ 与 $A_{2n+1}$ 共点 [See](https://codeforces.com/contest/24/problem/C) 。
- 已知 $A\ (x_A, y_A)$ 和 $X\ (x_X,y_X)$ 两点及这两点的坐标，构造 $Y$ 使得 $X,Y$ 关于 $A$ 对称，那么 $Y$ 的坐标为 $(2\cdot x_A-x_X,2\cdot y_A-y_X)$ 。
- **海伦公式**：已知三角形三边长 $a,b$ 和 $c$ ，定义 $p=\dfrac{a+b+c}{2}$ ，则 $S_{\triangle}=\sqrt{p(p-a)(p-b)(p-c)}$ ，在使用时需要注意越界问题，本质是铅锤定理，一般多使用叉乘计算三角形面积而不使用该公式。
- 棱台体积 $V=\frac{1}{3}(S_1+S_2+\sqrt{S_1S_2})\cdot h$，其中 $S_1,S_2$ 为上下底面积。
- 正棱台侧面积 $\frac{1}{2}(C_1+C_2)\cdot L$，其中 $C_1,C_2$ 为上下底周长，$L$ 为斜高（上下底对应的平行边的距离）。
- 球面积 $4\pi r^2$，体积 $\frac{4}{3}\pi r^3$。
- 正三角形面积 $\dfrac{\sqrt 3 a^2}{4}$，正四面体面积 $\dfrac{\sqrt 2 a^3}{12}$。
- 设扇形对应的圆心角弧度为 $\theta$ ，则面积为 $S=\frac{\theta}{2}\cdot R^2$ 。

### 5.14.2 立体几何结论归档

- 已知向量 $\vec{r}=\{x,y,z\}$ ，则该向量的三个方向余弦为 $\cos \alpha =\dfrac{x}{|\vec r|}=\dfrac{x}{\sqrt{x^2+y^2+z^2}}; \ \cos \beta = \dfrac{y}{|\vec r|};\ \cos \gamma =\dfrac{z}{|\vec r|}$ 。其中 $\alpha,\beta,\gamma\in [0,\pi]$ ，$\cos^2\alpha+\cos^2\beta+\cos^2\gamma=1$ 。

## 5.15 常用例题

### 5.15.1 将平面某点旋转任意角度

题意：给定平面上一点 $(a,b)$ ，输出将其逆时针旋转 $d$ 度之后的坐标。

```c++
signed main() {
    int a, b, d;
    cin >> a >> b >> d;
    
    ld l = hypot(a, b); // 库函数，求直角三角形的斜边
    ld alpha = atan2(b, a) + toArc(d);
    
    cout << l * cos(alpha) << " " << l * sin(alpha) << endl;
}
```

### 5.15.2 平面最近点对（set解）

借助 `set` ，在严格 $\mathcal O(N\log N)$ 复杂度内求解，比常见的分治法稍快。

```c++
template<class T> T sqr(T x) {
    return x * x;
}

using V = Point<int>;
signed main() {
    int n;
    cin >> n;

    vector<V> in(n);
    for (auto &it : in) {
        cin >> it;
    }

    int dis = disEx(in[0], in[1]); // 设定阈值
    sort(in.begin(), in.end());

    set<V> S;
    for (int i = 0, h = 0; i < n; i++) {
        V now = {in[i].y, in[i].x};
        while (dis && dis <= sqr(in[i].x - in[h].x)) { // 删除超过阈值的点
            S.erase({in[h].y, in[h].x});
            h++;
        }
        auto it = S.lower_bound(now);
        for (auto k = it; k != S.end() && sqr(k->x - now.x) < dis; k++) {
            dis = min(dis, disEx(*k, now));
        }
        if (it != S.begin()) {
            for (auto k = prev(it); sqr(k->x - now.x) < dis; k--) {
                dis = min(dis, disEx(*k, now));
                if (k == S.begin()) break;
            }
        }
        S.insert(now);
    }
    cout << sqrt(dis) << endl;
}
```

### 5.15.3 平面若干点能构成的最大四边形的面积（简单版，暴力枚举）

题意：平面上存在若干个点，保证没有两点重合、没有三点共线，你需要从中选出四个点，使得它们构成的四边形面积是最大的，注意这里能组成的四边形可以不是凸四边形。

暴力枚举其中一条对角线后枚举剩余两个点，$\mathcal O(N^3)$ 。

```c++
signed main() {
    int n;
    cin >> n;
    vector<Pi> in(n);
    for (auto &it : in) {
        cin >> it;
    }
    ld ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) { // 枚举对角线
            ld l = 0, r = 0;
            for (int k = 0; k < n; k++) { // 枚举第三点
                if (k == i || k == j) continue;
                if (pointOnLineLeft(in[k], {in[i], in[j]})) {
                    l = max(l, triangleS(in[k], in[j], in[i]));
                } else {
                    r = max(r, triangleS(in[k], in[j], in[i]));
                }
            }
            if (l * r != 0) { // 确保构成的是四边形
                ans = max(ans, l + r);
            }
        }
    }
    cout << ans << endl;
}
```

### 5.15.4 平面若干点能构成的最大四边形的面积（困难版，分类讨论+旋转卡壳）

题意：平面上存在若干个点，可能存在多点重合、共线的情况，你需要从中选出四个点，使得它们构成的四边形面积是最大的，注意这里能组成的四边形可以不是凸四边形、可以是退化的四边形。

当凸包大小 $\le 2$ 时，说明是退化的四边形，答案直接为 $0$ ；大小恰好为 $3$ 时，说明是凹四边形，我们枚举不在凸包上的那一点，将两个三角形面积相减既可得到答案；大小恰好为 $4$ 时，说明是凸四边形，使用旋转卡壳求解。

```c++
using V = Point<int>;
signed main() {
    int Task = 1;
    for (cin >> Task; Task; Task--) {
        int n;
        cin >> n;
        
        vector<V> in_(n);
        for (auto &it : in_) {
            cin >> it;
        }
        auto in = staticConvexHull(in_, 0);
        n = in.size();
        
        int ans = 0;
        if (n > 3) {
            ans = rotatingCalipers(in);
        } else if (n == 3) {
            int area = triangleAreaEx(in[0], in[1], in[2]);
            for (auto it : in_) {
                if (it == in[0] || it == in[1] || it == in[2]) continue;
                int Min = min({triangleAreaEx(it, in[0], in[1]), triangleAreaEx(it, in[0], in[2]), triangleAreaEx(it, in[1], in[2])});
                ans = max(ans, area - Min);
            }
        }
        
        cout << ans / 2;
        if (ans % 2) {
            cout << ".5";
        }
        cout << endl;
    }
}
```

### 5.15.5 线段将多边形切割为几个部分

题意：给定平面上一线段与一个任意多边形，求解线段将多边形切割为几个部分；保证线段的端点不在多边形内、多边形边上，多边形顶点不位于线段上，多边形的边不与线段重叠；多边形端点按逆时针顺序给出。下方的几个样例均合法，答案均为 $3$ 。

<img src="https://img2023.cnblogs.com/blog/2491503/202308/2491503-20230827211620035-1506522300.png" alt="截图" style="zoom:60%;" /><img src="https://img2023.cnblogs.com/blog/2491503/202308/2491503-20230827211624866-1953825694.png" alt="截图" style="zoom:60%;" />

当线段切割多边形时，本质是与多边形的边交于两个点、或者说是与多边形的两条边相交，设交点数目为 $x$ ，那么答案即为 $\frac{x}{2}+1$ 。于是，我们只需要计算交点数量即可，先判断某一条边是否与线段相交，再判断边的两个端点是否位于线段两侧。

```c++
signed main() {
    Pi s, e;
    cin >> s >> e; // 读入线段
    
    int n;
    cin >> n;
    vector<Pi> in(n);
    for (auto &it : in) {
        cin >> it; // 读入多边形端点
    }
    
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        Pi x = in[i], y = in[(i + 1) % n];
        cnt += (pointNotOnLineSide(x, y, {s, e}) && segmentIntersection(Line{x, y}, {s, e}));
    }
    cout << cnt / 2 + 1 << endl;
}
```

### 5.15.6 平面若干点能否构成凸包（暴力枚举）

题意：给定平面上若干个点，判断其是否构成凸包 [See](https://atcoder.jp/contests/abc266/tasks/abc266_c) 。

可以直接使用凸包模板，但是代码较长；在这里我们使用暴力枚举试点，也能以 $\mathcal O(N)$ 的复杂度通过。当两个向量的叉乘 $\le0$ 时说明其夹角大于等于 $180\degree$ ，使用这一点即可判定。

```c++
signed main() {
    int n;
    cin >> n;
    
    vector<Point<ld>> in(n);
    for (auto &it : in) {
        cin >> it;
    }
    
    for (int i = 0; i < n; i++) {
        auto A = in[(i - 1 + n) % n];
        auto B = in[i];
        auto C = in[(i + 1) % n];
        if (cross(A - B, C - B) > 0) {
            cout << "No\n";
            return 0;
        }
    }
    cout << "Yes\n";
}
```

### 5.15.7 凸包上的点能构成的最大三角形（暴力枚举）

可以直接使用凸包模板，但是代码较长；在这里我们使用暴力枚举试点，也能以 $\mathcal O(N)$ 的复杂度通过。

> 另外补充一点性质：所求三角形的反互补三角形一定包含了凸包上的所有点（可以在边界）。通俗的说，构成的三角形是这个反互补三角形的中点三角形。如下图所示，点 $A$ 不在 $\triangle BCE$ 的反互补三角形内部，故 $\triangle BCE$ 不是最大三角形；$\triangle ACE$ 才是。
> 
> <img src="https://img2023.cnblogs.com/blog/2491503/202308/2491503-20230827205516769-1055425260.png" alt="截图" style="zoom:80%;" />
> 
> ![](https://img2023.cnblogs.com/blog/2491503/202308/2491503-20230827205528116-1886683012.png)

```c++
signed main() {
    int n;
    cin >> n;
     
    vector<Point<int>> in(n);
    for (auto &it : in) {
        cin >> it;
    }
    
    #define S(x, y, z) triangleAreaEx(in[x], in[y], in[z])
     
    int i = 0, j = 1, k = 2;
    while (true) {
        int val = S(i, j, k);
        if (S((i + 1) % n, j, k) > val) {
            i = (i + 1) % n;
        } else if (S((i - 1 + n) % n, j, k) > val) {
            i = (i - 1 + n) % n;
        } else if (S(i, (j + 1) % n, k) > val) {
            j = (j + 1) % n;
        } else if (S(i, (j - 1 + n) % n, k) > val) {
            j = (j - 1 + n) % n;
        } else if (S(i, j, (k + 1) % n) > val) {
            k = (k + 1) % n;
        } else if (S(i, j, (k - 1 + n) % n) > val) {
            k = (k - 1 + n) % n;
        } else {
            break;
        }
    }
    cout << i + 1 << " " << j + 1 << " " << k + 1 << endl;
}
```

### 5.15.8 凸包上的点能构成的最大四角形的面积（旋转卡壳）

由于是凸包上的点，所以保证了四边形一定是凸四边形，时间复杂度 $\mathcal O(N^2)$ 。

```c++
template<class T> T rotatingCalipers(vector<Point<T>> &p) {
    #define S(x, y, z) triangleAreaEx(p[x], p[y], p[z])
    int n = p.size();
    T ans = 0;
    auto nxt = [&](int i) -> int {
        return i == n - 1 ? 0 : i + 1;
    };
    for (int i = 0; i < n; i++) {
        int p1 = nxt(i), p2 = nxt(nxt(nxt(i)));
        for (int j = nxt(nxt(i)); nxt(j) != i; j = nxt(j)) {
            while (nxt(p1) != j && S(i, j, nxt(p1)) > S(i, j, p1)) {
                p1 = nxt(p1);
            }
            if (p2 == j) {
                p2 = nxt(p2);
            }
            while (nxt(p2) != i && S(i, j, nxt(p2)) > S(i, j, p2)) {
                p2 = nxt(p2);
            }
            ans = max(ans, S(i, j, p1) + S(i, j, p2));
        }
    }
    return ans;
    #undef S
}
```

### 5.15.9 判断一个凸包是否完全在另一个凸包内

题意：给定一个凸多边形 $A$ 和一个凸多边形 $B$ ，询问 $B$ 是否被 $A$ 包含，分别判断严格/不严格包含。[例题](https://codeforces.com/contest/166/problem/B)。

考虑严格包含，使用 $A$ 点集计算出凸包 $T_1$ ，使用 $A,B$ 两个点集计算出不严格凸包 $T_2$ ，如果包含，那么 $T_1$ 应该与 $T_2$ 完全相等；考虑不严格包含，在计算凸包 $T_2$ 时严格即可。最终以 $\mathcal O(N)$ 复杂度求解，且代码不算很长。

# 6 图论

## 6.1 迪杰斯特拉 Dijkstra

```cpp
using pii = pair<int, int>;
using graph = vector<vector<pii>>;
constexpr int INF = 1e18;

vector<int> dijkstra(graph &g, int s) {
    vector<int> dist(g.size(), INF);
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    dist[s] = 0;
    pq.emplace(0, s);
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (dist[u] < d) continue;
        for (auto [w, v] : g[u]) {
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                pq.emplace(dist[v], v);
            }
        }
    }
    return dist;
}
```

## 6.2 判断负环 SPFA

用于判断负环是否存在，有负环 return true，否则 return false

```cpp
bool SPFA(int start, graph &g) {
    int n = (int)g.size();
    vector<int> dist(n, INT_MAX), in_cnt(n, 0);
    bitset<MAXN> vis; // vector<bool> vis(n);
    queue<int> q;
    dist[start] = 0;
    q.push(start);
    vis[start] = true;
    while (!q.empty()) {
        int now = q.front();
        q.pop();
        vis[now] = false;
        for (auto &[dis, to] : g[now]) {
            if (dist[now] + dis < dist[to]) {
                dist[to] = dist[now] + dis;
                if (!vis[to]) {
                    q.push(to);
                    in_cnt[to]++;
                    if (in_cnt[to] >= (int)g.size()) {
                        return true;
                    }
                    vis[to] = true;
                }
            }
        }
    }
    return false;
}
```

## 6.3 拓扑排序 TopoSort

输入图，入度，节点数，返回拓扑序

```cpp
using vvi = vector<vector<int>>;
vector<int> topoSort(vvi &g, vector<int> indeg) {
    queue<int> q;
    for (size_t i = 0; i < indeg.size(); ++i) {
        if (indeg[i] == 0) q.push(i);
    }
    vector<int> res;
    res.reserve(g.size());
    while (!q.empty()) {
        int u = q.front();
        res.emplace_back(u);
        q.pop();
        for (int v : g[u]) {
            indeg[v]--;
            if (indeg[v] == 0) {
                q.push(v);
            }
        }
    }
    assert(res.size() == g.size());
    return res;
}
```

## 6.4 强连通分量

缩点后，从tot到1就是拓扑序，无需再进行拓扑排序。

```cpp
// Luogu P3387
#include <bits/stdc++.h>
#define int long long
using namespace std;
using vvi = vector<vector<int>>;
constexpr int N = 1e4 + 5;
constexpr int M = 1e5 + 5;
int dfn[N], low[N], tim;
int stk[N], instk[N], tp;
void stkPush(int x) { stk[++tp] = x; instk[x] = 1; }
int stkPop() { int y = stk[tp--]; instk[y] = 0; return y; }
int id[N], tot;
void tarjan(int x, const vvi &g) {
    dfn[x] = low[x] = ++tim;
    stkPush(x);
    for (int y : g[x]) {
        if (!dfn[y]) tarjan(y, g), low[x] = min(low[x], low[y]);
        else if (instk[y]) low[x] = min(low[x], dfn[y]);
    }
    if (low[x] == dfn[x]) {
        tot++;
        while (stk[tp + 1] != x) id[stkPop()] = tot;
    }
}
int n, m, a[N], b[N], f[N], ans;  // b 为新图的点权，f为x新图中到点x的路径点权最大和
signed main() {
    ios::sync_with_stdio(false), cin.tie(nullptr);
    cin >> n >> m;
    for (int i = 1; i <= n; i++) cin >> a[i];
    vvi g(n + 1);
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        g[u].emplace_back(v);
    }
    for (int i = 1; i <= n; i++)
        if (!dfn[i]) tarjan(i, g);
    vvi h(tot + 1);
    for (int i = 1; i <= n; i++) {
        for (int j : g[i]) { // 缩点
            int x = id[i], y = id[j];
            if (x != y) h[x].emplace_back(y);
        }
    }
    for (int i = 1; i <= n; i++) b[id[i]] += a[i];
    for (int i = 1; i <= tot; i++) { // DAG dp
        f[i] = b[i];
        for (int j : h[i]) f[i] = max(f[i], b[i] + f[j]);
        ans = max(ans, f[i]);
    }
    cout << ans << "\n";
}
```

## 6.5 点双连通分量+割点

```cpp
// Luogu P8435
#include <bits/stdc++.h>
#define int long long
using namespace std;
using vvi = vector<vector<int>>;
constexpr int N = 5e5 + 5;
constexpr int M = 2e6 + 5;
int dfn[N], low[N], tim;
int stk[N], tp;
vvi dcc(N);
int n, m, tot;
bool cut[N];
void tarjan(int x, int fa, const vvi &g) {
    dfn[x] = low[x] = ++tim;
    stk[++tp] = x;
    int child = 0; 
    for (int y : g[x]) {
        if (!dfn[y]) {
            tarjan(y, x, g), low[x] = min(low[x], low[y]);
            if (low[y] >= dfn[x]) {
                child++, tot++;
                if (fa != 0 or child > 1) cut[x] = true;
                while (stk[tp+1] != y) dcc[tot].emplace_back(stk[tp--]);
                dcc[tot].emplace_back(x);
            }
        } else if (y != fa)
            low[x] = min(low[x], dfn[y]);
    }
    if (fa == 0 and child == 0) dcc[++tot].emplace_back(x);  
}
signed main() {
    ios::sync_with_stdio(false), cin.tie(nullptr);
    cin >> n >> m;
    vvi g(n + 1);
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        g[u].emplace_back(v), g[v].emplace_back(u);
    }
    for (int i = 1; i <= n; i++)
        if (!dfn[i]) tarjan(i, 0, g);
    cout << tot << "\n";
    for (int i = 1; i <= tot; i++) {
        cout << dcc[i].size();
        for (int j : dcc[i]) cout << " " << j;
        cout << "\n";
    }
}
```

## 6.6 边双连通分量+割边

```cpp
// Luogu P8436
#include <bits/stdc++.h>
#define int long long
using namespace std;
constexpr int N = 5e5 + 5;
constexpr int M = 2e6 + 5;
struct Edge {
    int to, nex;
} e[M * 2];
int head[N], cnt = 1;
void add_edge(int x, int y) {
    e[++cnt] = {y, head[x]};
    head[x] = cnt;
}
int dfn[N], low[N], tim;
int stk[N], tp;
int id[N], tot, bri[M * 2];
void tarjan(int x, int eid) {
    dfn[x] = low[x] = ++tim;
    stk[++tp] = x;
    for (int i = head[x]; i; i = e[i].nex) {
        int y = e[i].to;
        if (!dfn[y]) {
            tarjan(y, i), low[x] = min(low[x], low[y]);
            bri[i] = bri[i ^ 1] = (low[y] > dfn[x]);
        } else if (i != (eid ^ 1))
            low[x] = min(low[x], dfn[y]);
    }
    if (low[x] == dfn[x]) {
        ++tot;
        while (stk[tp + 1] != x) id[stk[tp--]] = tot;
    }
}
signed main() {
    ios::sync_with_stdio(false), cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        add_edge(u, v), add_edge(v, u);
    }
    for (int i = 1; i <= n; i++)
        if (!dfn[i]) tarjan(i, 0);
    cout << tot << "\n";
    vector<vector<int>> dcc(tot + 1);
    for (int i = 1; i <= n; i++) 
        dcc[id[i]].emplace_back(i);
    for (int i = 1; i <= tot; i++) {
        cout << dcc[i].size();
        for (int j : dcc[i]) cout << " " << j;
        cout << "\n";
    }
}
```

## 6.7 最大流

Dinic多路增广。初始正向边c，反向边0。

```cpp
constexpr int N = 1000005;
constexpr int M = 1000005;
constexpr int INF = 1e18;
int n, m, S, T, ans, tot; // tot 为总的结点个数
struct Edge {int to, c, nex;} e[M*2];
int head[N], cnt = 1;
void add(int u, int v, int c) {
    e[++cnt] = {v, c, head[u]};
    head[u] = cnt;
}
void add_edge(int u, int v, int c) {
    add(u, v, c); add(v, u, 0);
}
int dep[N];
bool bfs() { // 分层
    fill(dep, dep+1+tot, 0);
    queue<int> q;
    q.push(S); dep[S] = 1; 
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int i = head[u]; i; i = e[i].nex) {
            int v = e[i].to;
            if (!dep[v] and e[i].c) {
                dep[v] = dep[u] + 1;
                q.push(v);
            }
        }
    }
    return dep[T] != 0;
}
int cur[N];
int dfs(int u, int fin) { // 多路增广
    if (u == T) return fin;
    int fout = 0;
    for (int &i = cur[u]; i and fin; i = e[i].nex) {
        int v = e[i].to;
        if (dep[v] > dep[u] and e[i].c) {
            int f = dfs(v, min(fin, e[i].c));
            fin -= f, fout += f;
            e[i].c -= f, e[i^1].c += f;
        }
    }
    if (fout == 0) dep[u] = 0;
    return fout;
}
int dinic() {
    int flow = 0;
    while (bfs()) {
        copy(head, head+1+tot, cur);
        flow += dfs(S, INF);
    }
    return flow;
}
```

## 6.8 费用流

最小费用最大流，SPFA+Dinic。

```cpp
constexpr int N = 5e3 + 5;
constexpr int M = 5e4 + 5;
constexpr int INF = 1e9;
int n, m, S, T;
struct Edge {
    int to, cap, cost, nex;
} e[M*2];
int head[N], cnt = 1, tot;
void add_edge(int u, int v, int cap, int cost) {
    e[++cnt] = {v, cap, cost, head[u]};
    head[u] = cnt;
}
int dist[N], in[N];
bool spfa() {
    fill(dist, dist+1+tot, INF);
    fill(in, in+1+tot, 0);
    queue<int> q;
    q.emplace(S); dist[S] = 0; in[S] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop(); in[u] = 0;
        for (int i = head[u]; i; i = e[i].nex) {
            int v = e[i].to;
            if (dist[u] + e[i].cost < dist[v] and e[i].cap) {
                dist[v] = dist[u] + e[i].cost;
                if (!in[v]) q.push(v), in[v] = 1;
            }
        }
    }
    return dist[T] != INF;
}
int cur[N];
int dfs(int u, int fin) {
    if (u == T) return fin;
    int fout = 0;
    in[u] = 1;
    for (int &i = cur[u]; i and fin; i = e[i].nex) {
        int v = e[i].to;
        if (!in[v] and dist[v] == dist[u] + e[i].cost and e[i].cap) {
            int f = dfs(v, min(fin, e[i].cap));
            fin -= f, fout += f;
            e[i].cap -= f, e[i^1].cap += f;
        }
    }
    in[u] = 0;
    if (fout == 0) dist[u] = 0;
    return fout;
}
pair<int, int> dinic() {
    int flow = 0, cost = 0;
    while (spfa()) {
        copy(head, head+1+tot, cur);
        int f = dfs(S, INF);
        flow += f, cost += f * dist[T];
    }
    return {flow, cost};
}
```

# 7 树论

## 7.1 树上倍增

LCA、树上路径、树上路径边权最值。

```cpp
constexpr int N = 1e5 + 5;
int dep[N], fa[20][N]; // 2^20 = 1048576
vector<vector<int>> g(N);
void dfs(int x, int f) {
    dep[x] = dep[f] + 1, fa[0][x] = f;
    for (int i = 1; i < 20; ++i)
        fa[i][x] = fa[i-1][fa[i-1][x]];
    for (int y : g[x])
        if (y != f) dfs(y, x);
}
int lca(int x, int y) {
    if (dep[x] < dep[y]) swap(x, y);
    for (int i = 19; ~i; --i)
        if (dep[fa[i][x]] >= dep[y]) x = fa[i][x];
    if (x == y) return y;
    for (int i = 19; ~i; --i)
        if (fa[i][x] != fa[i][y])
            x = fa[i][x], y = fa[i][y];
    return fa[0][x];
}
```

## 7.2 dfs序LCA

$O(n\log{n})$ 预处理，$O(1)$ 查询。

```cpp
#include <bits/stdc++.h>
using namespace std;
constexpr int N = 5e5 + 5;
constexpr int K = __lg(N) + 3;
int n, m, s;
int dfn[N], cnt, st[K][N]; 
vector<vector<int>> g(N);
void dfs(int x, int f) {
    dfn[x] = ++cnt;
    st[0][dfn[x]] = f;
    for (int y : g[x])
        if (y != f) dfs(y, x);
}
void build_lca() {
    for (int j = 1; j < K; j++) {
        for (int i = 1; i + (1 << j) - 1 <= n; i++) {
            int p1 = st[j-1][i];
            int p2 = st[j-1][i + (1 << (j-1))];
            st[j][i] = dfn[p1] < dfn[p2] ? p1 : p2;
        }
    }
}
int lca(int x, int y) {
    if (x == y) return x;
    if (dfn[x] > dfn[y]) swap(x, y);
    int j = __lg(dfn[y] - dfn[x]);
    int p1 = st[j][dfn[x]+1];
    int p2 = st[j][dfn[y]-(1<<j)+1];
    return dfn[p1] < dfn[p2] ? p1 : p2; 
}
signed main() {
    dfs(s, 0); // s 是根
    build_lca();
    cout << lca(u, v);
}

```

## 7.3 虚树

当查询点总数有限时，根据查询点及其LCA建立虚树，然后在虚树上面dp。**需要预处理LCA和dfs序**。

```cpp
vector<int> g[N], h[N], used; // 原树、虚树、虚树中用到的点
int stk[N], top;

void add_edge(int x, int y) {
    h[x].emplace_back(y);
    used.emplace_back(x);
}

void clear() {
    for (int x : used) {
        h[x].clear();
    }
    used.clear();
}

void build() {
    sort(a+1,a+1+k,[&](int x, int y) {
        return dfn[x] < dfn[y];
    });
    stk[top=1] = 1;
    if (a[1] != 1) stk[++top] = a[1];
    for (int i = 2; i <= k; i++) {
        int c = lca(stk[top], a[i]);
        while (top > 1 and dep[stk[top-1]] >= dep[c]) {
            add_edge(stk[top-1], stk[top]);
            top--;
        }        
        if (c != stk[top]) {
            add_edge(c, stk[top]);
            stk[top] = c;
        }
        stk[++top] = a[i];
    }
    while (top > 1) {
        add_edge(stk[top-1], stk[top]);
        top--;
    }
}
```

# 8 数据结构

## 8.1 扩展树状数组

注意，被查询的值都应该小于等于 $N$ ，否则会越界；如果离散化不可使用，则需要使用平衡树替代。

```cpp
struct BIT {
    int n;
    vector<int> w;
    BIT(int n) : n(n), w(n + 1) {}
    void add(int x, int v) {
        for (; x <= n; x += x & -x) {
            w[x] += v;
        }
    }
    int kth(int x) { // 查找第 k 小的值
        int ans = 0;
        for (int i = __lg(n); i >= 0; i--) {
            int val = ans + (1 << i);
            if (val < n && w[val] < x) {
                x -= w[val];
                ans = val;
            }
        }
        return ans + 1;
    }
    int get(int x) { // 查找 x 的排名
        int ans = 1;
        for (x--; x; x -= x & -x) {
            ans += w[x];
        }
        return ans;
    }
    int pre(int x) { return kth(get(x) - 1); } // 查找 x 的前驱
    int suf(int x) { return kth(get(x + 1)); } // 查找 x 的后继
};
const int N = 10000000; // 可以用于在线处理平衡二叉树的全部要求
signed main() {
    BIT bit(N + 1); // 在线处理不能够离散化，一定要开到比最大值更大
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        int op, x;
        cin >> op >> x;
        if (op == 1) bit.add(x, 1); // 插入 x
        else if (op == 2) bit.add(x, -1); // 删除任意一个 x
        else if (op == 3) cout << bit.get(x) << "\n"; // 查询 x 的排名
        else if (op == 4) cout << bit.kth(x) << "\n"; // 查询排名为 x 的数
        else if (op == 5) cout << bit.pre(x) << "\n"; // 求小于 x 的最大值（前驱）
        else if (op == 6) cout << bit.suf(x) << "\n"; // 求大于 x 的最小值（后继）
    }
}
```

## 8.2 可持久化线段树（主席树）

单点改，单点查。

```cpp
#include <bits/stdc++.h>
#define int long long
#define mid ((l+r)>>1)
using namespace std;
constexpr int N = 1e6 + 5;
constexpr int M = 25 * N;
int a[N], n, m;
int ls[M], rs[M], val[M];
int root[N], tot;
void build(int &u, int l, int r) {
    u = ++tot;
    if (l == r) {
        val[u] = a[l];
        return;
    }
    build(ls[u], l, mid);
    build(rs[u], mid+1, r);
}
void change(int &u, int v, int l, int r, int p, int c) {
    u = ++tot;
    ls[u] = ls[v], rs[u] = rs[v], val[u] = val[v];
    if (l == r) {
        val[u] = c;
        return;
    }
    if (p <= mid) change(ls[u], ls[v], l, mid, p, c);
    else change(rs[u], rs[v], mid+1, r, p, c);
}
int query(int u, int l, int r, int p) {
    if (l == r) return val[u];
    if (p <= mid) return query(ls[u], l, mid, p);
    else return query(rs[u], mid+1, r, p);
}
signed main() {
    ios::sync_with_stdio(false), cin.tie(nullptr);
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    build(root[0], 1, n);
    for (int i = 1; i <= m; i++) {
        int v, op, p, c;
        cin >> v >> op >> p;
        if (op == 1) {
            cin >> c;
            change(root[i], root[v], 1, n, p, c);
        } else if (op == 2) {
            root[i] = root[v];
            cout << query(root[i], 1, n, p) << "\n";
        }
    }
}
```

单点改，二分查（静态区间第k小）

```cpp
#include <bits/stdc++.h>
#define int long long
#define ls(u) w[u].l
#define rs(u) w[u].r
#define mid ((l+r)>>1)
using namespace std;
constexpr int N = 1e6 + 5;
int n, m, a[N], b[N];
struct Node {
    int l, r, sum;
} w[N*25];
int root[N], tot;
void pushup(int u) {
    w[u].sum = w[ls(u)].sum + w[rs(u)].sum;
}
void change(int &u, int v, int l, int r, int p) {
    u = ++tot;
    w[u] = w[v];
    if (l == r) {
        w[u].sum++;
        return;
    }
    if (p <= mid) change(ls(u), ls(v), l, mid, p);
    else change(rs(u), rs(v), mid+1, r, p);
    pushup(u);
}
int query(int L, int R, int l, int r, int k) {
    int sum = w[ls(R)].sum - w[ls(L)].sum;
    if (l == r) return l;
    if (sum >= k) return query(ls(L), ls(R), l, mid, k);
    else return query(rs(L), rs(R), mid+1, r, k - sum);
}
signed main() {
    ios::sync_with_stdio(false), cin.tie(nullptr);
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        b[i] = a[i];
    }
    sort(b+1,b+1+n);
    auto ed = unique(b+1,b+1+n);
    for (int i = 1; i <= n; i++) {
        int t = lower_bound(b+1,ed,a[i]) - b;
        // cerr << t << " ";
        change(root[i], root[i-1], 1, n, t);
    }
    // cerr << "\n";
    for (int i = 1; i <= m; i++) {
        int L, R, k;
        cin >> L >> R >> k;
        int idx = query(root[L-1], root[R], 1, n, k);
        // cerr << idx << "\n";
        cout << b[idx] << "\n";
    }
}
```

## 8.3 重链剖分 (by Jiangly)

```cpp
struct HLD {
    int n;
    vector<int> siz, top, dep, parent, in, out, seq;
    vector<vector<int>> adj;
    int cur;
    
    HLD() {}
    HLD(int n) {
        init(n);
    }
    void init(int n) {
        this->n = n;
        siz.resize(n);
        top.resize(n);
        dep.resize(n);
        parent.resize(n);
        in.resize(n);
        out.resize(n);
        seq.resize(n);
        cur = 0;
        adj.assign(n, {});
    }
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void work(int root = 0) {
        top[root] = root;
        dep[root] = 0;
        parent[root] = -1;
        dfs1(root);
        dfs2(root);
    }
    void dfs1(int u) {
        if (parent[u] != -1) {
            adj[u].erase(find(adj[u].begin(), adj[u].end(), parent[u]));
        }
        
        siz[u] = 1;
        for (auto &v : adj[u]) {
            parent[v] = u;
            dep[v] = dep[u] + 1;
            dfs1(v);
            siz[u] += siz[v];
            if (siz[v] > siz[adj[u][0]]) {
                swap(v, adj[u][0]);
            }
        }
    }
    void dfs2(int u) {
        in[u] = cur++;
        seq[in[u]] = u;
        for (auto v : adj[u]) {
            top[v] = v == adj[u][0] ? top[u] : v;
            dfs2(v);
        }
        out[u] = cur;
    }
    int lca(int u, int v) {
        while (top[u] != top[v]) {
            if (dep[top[u]] > dep[top[v]]) {
                u = parent[top[u]];
            } else {
                v = parent[top[v]];
            }
        }
        return dep[u] < dep[v] ? u : v;
    }
    
    int dist(int u, int v) {
        return dep[u] + dep[v] - 2 * dep[lca(u, v)];
    }
    
    int jump(int u, int k) {
        if (dep[u] < k) {
            return -1;
        }
        
        int d = dep[u] - k;
        
        while (dep[top[u]] > d) {
            u = parent[top[u]];
        }
        
        return seq[in[u] - dep[u] + d];
    }
    
    bool isAncester(int u, int v) {
        return in[u] <= in[v] && in[v] < out[u];
    }
    
    int rootedParent(int u, int v) {
        swap(u, v);
        if (u == v) {
            return u;
        }
        if (!isAncester(u, v)) {
            return parent[u];
        }
        auto it = upper_bound(adj[u].begin(), adj[u].end(), v, [&](int x, int y) {
            return in[x] < in[y];
        }) - 1;
        return *it;
    }
    
    int rootedSize(int u, int v) {
        if (u == v) {
            return n;
        }
        if (!isAncester(v, u)) {
            return siz[v];
        }
        return n - siz[rootedParent(u, v)];
    }
    
    int rootedLca(int a, int b, int c) {
        return lca(a, b) ^ lca(b, c) ^ lca(c, a);
    }
};
```

## 8.4 ST表

```cpp
struct ST {
    vector<vector<int>> st;
    ST (const vector<int>& arr) {
        int n = arr.size();
        st.assign(__lg(n) + 1, vector<int>(n)); // 维度小的放前面
        for (int i = 0; i < n; ++i) st[0][i] = arr[i];
        for (int j = 1; 1 << j <= n; ++j) {
            for (int i = 0; i + (1 << j) <= n; ++i) {
                st[j][i] = max(st[j - 1][i], st[j - 1][i + (1 << (j - 1))]);
            }
        }
    }
    int query(int l, int r) { // [l, r]
        int k = __lg(r - l + 1);
        return max(st[k][l], st[k][r - (1 << k) + 1]);
    }
};
```

## 8.5 字典树 Trie

字符串的插入和查询，01字典树维护异或极值

```cpp
class Trie
{
private:
    int amount;
    vector<vector<int>> son;
    vector<int> cnt, prefix_cnt;
    // flag为true表示 查询该字符串数量
    // flag为false表示 查询以该字符串为前缀的字符串的数量
    int query(const string &str, bool flag)
    {
        int p = 0, image;
        for (auto ch : str)
        {
            image = f(ch);
            if (!son[p][image])
                return 0;
            p = son[p][image];
        }
        return flag ? cnt[p] : prefix_cnt[p];
    }
    function<int(char)> f;

public:
    // f的返回值应该在[0, n)范围内
    Trie(function<int(char)> function, int n) : son(1, vector<int>(n)), cnt(1), prefix_cnt(1) { f = function, amount = n; }
    void insert(const string &str)
    {
        int p = 0, image;
        for (auto ch : str)
        {
            image = f(ch);
            if (!son[p][image])
            {
                son[p][image] = son.size();
                vector<int> temp(amount);
                son.push_back(temp), cnt.push_back(0), prefix_cnt.push_back(0);
            }
            p = son[p][image];
            prefix_cnt[p]++;
        }
        cnt[p]++;
    }
    int queryAmount(const string &str) { return query(str, true); }
    int queryPrefixAmount(const string &str) { return query(str, false); }
};

int f(char ch)
{
    if (ch >= 'a' && ch <= 'z')
        return ch - 'a';
    else if (ch >= 'A' && ch <= 'Z')
        return ch - 'A' + 26;
    assert(ch >= '0' && ch <= '9');
    return ch - '0' + 52;
}
```

## 8.6 根号分块

```cpp
struct Sqrt {
	int block_size;
	vector<int> nums;
	vector<long long> blocks;
	Sqrt(int sqrtn, vector<int> &arr) : block_size(sqrtn), blocks(sqrtn, 0) {
		nums = arr;
		for (int i = 0; i < nums.size(); i++) { blocks[i / block_size] += nums[i]; }
	}

	/** O(1) update to set nums[x] to v */
	void update(int x, int v) {
		blocks[x / block_size] -= nums[x];
		nums[x] = v;
		blocks[x / block_size] += nums[x];
	}

	/** O(sqrt(n)) query for sum of [0, r) */
	long long query(int r) {
		long long res = 0;
		for (int i = 0; i < r / block_size; i++) { res += blocks[i]; }
		for (int i = (r / block_size) * block_size; i < r; i++) { res += nums[i]; }
		return res;
	}

	/** O(sqrt(n)) query for sum of [l, r) */
	long long query(int l, int r) { return query(r) - query(l - 1); }
};
```

## 8.7 莫队

如果修改可以做到 $O(1)$ 并且可以离线查询的话可以考虑，复杂度 $O(n \sqrt n)$

```cpp
// 询问区间内有多少个不同的数
int cmp(query a, query b) {
	return (belong[a.l] ^ belong[b.l]) ? belong[a.l] < belong[b.l] : 
    ((belong[a.l] & 1) ? a.r < b.r : a.r > b.r);
}
void add(int pos) {
    if(!cnt[aa[pos]]) ++now;
    ++cnt[aa[pos]];
}
void del(int pos) {
    --cnt[aa[pos]];
    if(!cnt[aa[pos]]) --now;
}
sort(q + 1, q + m + 1, cmp);
int l = 1, r = 0;
for(int i = 1; i <= q; ++i) {//对于每次询问
        int ql, qr;
        scanf("%d%d", &ql, &qr);//输入询问的区间
        while(l < ql) del(l++);//如左指针在查询区间左方，左指针向右移直到与查询区间左端点重合
        while(l > ql) add(--l);//如左指针在查询区间左端点右方，左指针左移
        while(r < qr) add(++r);//右指针在查询区间右端点左方，右指针右移
        while(r > qr) del(r--);//否则左移
        printf("%d\n", now);//输出统计结果
    }
}
```

## 8.8 笛卡尔树

这是一种键满足平衡二叉树性质，值满足堆性质的二叉树，用于某些序列上的计数问题。

```cpp
int stk[N], tp;
int p[N], ls[N], rs[N], n;
void build_tree() { // 小根堆
    for (int i = 1; i <= n; i++) {
        int j = 0;
        while (tp and p[stk[tp]] > p[i]) j = stk[tp--];
        ls[i] = j;
        if (tp) rs[stk[tp]] = i;
        stk[++tp] = i;
    }
}
```

## 8.9 可撤销并查集 (by Jiangly)

放弃路径压缩，保留按秩合并。find复杂度为 $O(\log n)$，回溯revert复杂度为 $O(k)$。用于线段树分治。

```cpp
struct DSU {O(k)
    vector<int> siz;
    vector<int> f;
    vector<array<int, 2>> his;
    
    DSU(int n) : siz(n + 1, 1), f(n + 1) {
        iota(f.begin(), f.end(), 0);
    }
    
    int find(int x) {
        while (f[x] != x) {
            x = f[x];
        }
        return x;
    }
    
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        if (siz[x] < siz[y]) {
            swap(x, y);
        }
        his.push_back({x, y});
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }
    
    int time() {
        return his.size();
    }
    
    void revert(int tm) {
        while (his.size() > tm) {
            auto [x, y] = his.back();
            his.pop_back();
            f[y] = y;
            siz[x] -= siz[y];
        }
    }
};
```

## 8.10 李超线段树

插入线段，查询某一点处的最小值。

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

const ll INF_LL = (ll)4e18;     // 用作“正无穷”，最小值模板使用正无穷作为占位
const ll X_L = -1000000000LL;   // 查询 x 的左端点（根据题意调整）
const ll X_R =  1000000000LL;   // 查询 x 的右端点（根据题意调整）

// -----------------
// 线： y = m*x + b
// -----------------
struct Line {
    ll m, b;
    // 默认构造使用 b = +INF（最小值模板）
    Line(ll _m = 0, ll _b = INF_LL) : m(_m), b(_b) {}

    // 评估函数，用 __int128 防止溢出
    ll eval(ll x) const {
        __int128 t = (__int128)m * x + (__int128)b;
        if (t > (__int128)INF_LL) return INF_LL;
        if (t < -(__int128)INF_LL) return -INF_LL;
        return (ll)t;
    }
};

/* =============================
   节点结构与根指针（动态节点）
   ============================= */
struct Node {
    Line ln;
    Node *l = nullptr, *r = nullptr;
    Node(const Line& _ln) : ln(_ln), l(nullptr), r(nullptr) {}
};

Node* root = nullptr;

// -----------------
// 在区间 [l,r] 的子树 node 中插入一条线（全域/段插入会调用它）
// 最小值版：比较使用 '<'
// -----------------
void add_line(Line nw, Node*& node, ll l = X_L, ll r = X_R){
    if (!node){
        node = new Node(nw);
        return;
    }
    ll mid = (l + r) >> 1;
    // ---------- 如果要改成最大值版：把下面两处 '<' 改为 '>' ----------
    bool lef = nw.eval(l) < node->ln.eval(l);    // <-- change '<' -> '>' for max
    bool m   = nw.eval(mid) < node->ln.eval(mid); // <-- change '<' -> '>' for max
    // --------------------------------------------------------------------
    if (m) swap(nw, node->ln);
    if (l == r) return;
    if (lef != m) add_line(nw, node->l, l, mid);
    else add_line(nw, node->r, mid + 1, r);
}

// wrapper：全局插线
inline void add_line(Line ln){
    add_line(ln, root, X_L, X_R);
}

// 在区间 [Lq,Rq] 上插入线段（线只在该区间生效）
void add_segment(Line nw, Node*& node, ll Lq, ll Rq, ll l = X_L, ll r = X_R){
    if (Rq < l || r < Lq) return;
    if (Lq <= l && r <= Rq){
        add_line(nw, node, l, r);
        return;
    }
    ll mid = (l + r) >> 1;
    // 占位节点：最小值版使用默认 Line(0, +INF)
    if (!node) node = new Node(Line(0, INF_LL)); // <-- 如果改为最大值版，这里要把 INF_LL 换成 NEG_INF（见下）
    add_segment(nw, node->l, Lq, Rq, l, mid);
    add_segment(nw, node->r, Lq, Rq, mid + 1, r);
}
inline void add_segment(Line nw, ll Lq, ll Rq){
    add_segment(nw, root, Lq, Rq, X_L, X_R);
}

// 查询点 x 的最小值（若节点为空返回 INF）
ll query(ll x, Node* node, ll l = X_L, ll r = X_R){
    if (!node) return INF_LL; // <-- 如果改为最大值版，这里要返回 NEG_INF
    ll res = node->ln.eval(x);
    if (l == r) return res;
    ll mid = (l + r) >> 1;
    if (x <= mid) {
        // ---------- 改为最大值时把 min -> max ----------
        return min(res, query(x, node->l, l, mid)); // <-- change to max(...) for max-version
    } else {
        return min(res, query(x, node->r, mid + 1, r)); // <-- change to max(...) for max-version
    }
}
inline ll query(ll x){
    return query(x, root, X_L, X_R);
}

// 释放树（若需要）
void clear_tree(Node* node){
    if (!node) return;
    clear_tree(node->l);
    clear_tree(node->r);
    delete node;
}

// -----------------
// 示例 main（演示用法）
// -----------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 示例：插入三条直线（最小值查询）
    add_line(Line(2, 3));    // y = 2x + 3
    add_line(Line(-1, 10));  // y = -x + 10
    add_line(Line(0, 5));    // y = 5

    cout << "query(0) = " << query(0) << "\n";   // 期望 min(3,10,5) = 3
    cout << "query(2) = " << query(2) << "\n";   // 期望 min(7,8,5) = 5
    cout << "query(10) = " << query(10) << "\n"; // 期望 min(23,0,5) = 0

    // 在区间 [1, 5] 插入一条只在此区间生效的线
    add_segment(Line(-3, 50), 1, 5); // y = -3x + 50 在 [1,5] 有效
    cout << "query(3) = " << query(3) << "\n";

    // 程序结束前释放内存（比赛中一般不需要）
    clear_tree(root);
    root = nullptr;
    return 0;
}
```

## 8.11 平衡树 fhq-Treap

### 8.11.1 按值分裂

普通平衡树，维护集合。

```cpp
#include <bits/stdc++.h>
using namespace std;
mt19937 rnd(chrono::high_resolution_clock::now().time_since_epoch().count());
struct Treap {
    static const int N = 2000000 + 5;
    int ls[N], rs[N], key[N], val[N], siz[N], root = 0, tot = 0, T1, T2, T3;
    int newNode(int v) {
        int u = ++tot;
        ls[u] = rs[u] = 0;
        key[u] = rnd();
        val[u] = v;
        siz[u] = 1;
        return u;
    }
    void push_up(int u) { siz[u] = siz[ls[u]] + siz[rs[u]] + 1; }
    int merge(int x, int y) {
        if (!x or !y) return x + y;
        if (key[x] > key[y]) {
            rs[x] = merge(rs[x], y);
            push_up(x);
            return x;
        } else {
            ls[y] = merge(x, ls[y]);
            push_up(y);
            return y;
        }
    }
    void split(int u, int v, int &x, int &y) {
        if (!u) {
            x = y = 0;
            return;
        }
        if (val[u] <= v) {
            x = u;
            split(rs[u], v, rs[u], y);
        } else {
            y = u;
            split(ls[u], v, x, ls[u]);
        }
        push_up(u);
    }
    void insert(int v) {
        split(root, v, T1, T2);
        root = merge(merge(T1, newNode(v)), T2);
    }
    void erase(int v) {
        split(root, v - 1, T1, T2);
        split(T2, v, T2, T3);
        root = merge(merge(T1, merge(ls[T2], rs[T2])), T3);
    }
    int rank(int v) {
        split(root, v - 1, T1, T2);
        int r = siz[T1] + 1;
        root = merge(T1, T2);
        return r;
    }
    int kth(int k) {
        int u = root;
        while (u) {
            int s = siz[ls[u]] + 1;
            if (k == s) break;
            if (k < s)
                u = ls[u];
            else
                k -= s, u = rs[u];
        }
        return val[u];
    }
    int pre(int v) {
        int u = root, best = INT_MIN;
        while (u) {
            if (val[u] < v)
                best = max(best, val[u]), u = rs[u];
            else
                u = ls[u];
        }
        return best;
    }
    int nex(int v) {
        int u = root, best = INT_MAX;
        while (u) {
            if (val[u] > v)
                best = min(best, val[u]), u = ls[u];
            else
                u = rs[u];
        }
        return best;
    }
};
```

### 8.11.2 按排名分裂

文艺平衡树，维护序列，这里给出的是实现序列的区间翻转操作。

```cpp
#include <bits/stdc++.h>
using namespace std;
constexpr int N = 1e5 + 5;
mt19937 rnd(time(nullptr));
struct Treap {
    int ls[N], rs[N], key[N], val[N], siz[N];
    int root = 0, tot = 0, T1, T2, T3;
    bool tag[N];
    int node(int v) {
        int u = ++tot;
        ls[u] = rs[u] = 0;
        key[u] = rnd();
        val[u] = v;
        siz[u] = 1;
        return u;
    }
    void pushup(int u) {
        siz[u] = siz[ls[u]] + siz[rs[u]] + 1;
    }
    void pushdown(int u) {
        if (tag[u]) {
            swap(ls[u], rs[u]);
            if (ls[u]) tag[ls[u]] ^= 1;
            if (rs[u]) tag[rs[u]] ^= 1;
            tag[u] = false;
        }
    }
    void split(int u, int k, int &x, int &y) {
        if (!u) { x = y = 0; return;}
        pushdown(u);
        if (siz[ls[u]] >= k) {
            y = u;
            split(ls[u], k, x, ls[u]);
        } else {
            x = u;
            split(rs[u], k - siz[ls[u]] - 1, rs[u], y);
        }
        pushup(u);
    }
    int merge(int x, int y) {
        if (!x or !y) return x + y;
        if (key[x] <= key[y]) {
            pushdown(y);
            ls[y] = merge(x, ls[y]);
            pushup(y);
            return y;
        } else {
            pushdown(x);
            rs[x] = merge(rs[x], y);
            pushup(x);
            return x;
        }
    }
    void reverse(int l, int r) {
        split(root, l - 1, T1, T2);
        split(T2, r - l + 1, T2, T3);
        tag[T2] ^= true;
        root = merge(T1, merge(T2, T3));
    }
    void output(int u) {
        if (u == 0) return;
        pushdown(u);
        output(ls[u]);
        cout << val[u] << " ";
        output(rs[u]);
    }
} s;
int main() {
    ios::sync_with_stdio(false), cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        s.root = s.merge(s.root, s.node(i));
    }
    while (m--) {
        int l, r; cin >> l >> r;
        s.reverse(l, r);
    }
    s.output(s.root);
}
```

# 9 字符串

## 9.1 字符串哈希

```cpp
constexpr int MOD = 1610612741;  // 212370440130137957LL 这个数虽大但是乘法会溢出
constexpr int base = 131;

// 单哈希法获得一个字符串的哈希值
ull getHash(string &s) {
    ull res = 0;
    for (int i = 0; i < s.length(); i++) {
        res = (base * res + (ull)s[i]) % MOD;
    }
    return res;
}
```

## 9.2 区间回文查询

依靠字符串哈希值的前缀和与后缀和

```cpp
using ull = unsigned long long;
constexpr int MAXN = 2E5 + 5;
constexpr ull MOD = 1610612741;
constexpr ull base = 131;
class MyStr {
   private:
    int n;
    char s[MAXN];
    ull pre[MAXN], suf[MAXN];
    void init() {
        pwb[0] = 1;
        for (int i = 1; i < MAXN; i++) {
            pwb[i] = pwb[i - 1] * base % MOD;
        }
    }
    ull hashL(int l, int r) { return (pre[r] - pre[l - 1] * pwb[r - l + 1] % MOD + MOD) % MOD; }
    ull hashR(int l, int r) { return (suf[l] - suf[r + 1] * pwb[r - l + 1] % MOD + MOD) % MOD; }

   public:
    ull pwb[MAXN];  // power of base : pwb[i]即base的i次幂
    MyStr() { init(); }
    // 建立哈希，每次读入新的字符串就要调用一下以更新哈希值
    void build(int n, char str[]) {
        pre[0] = suf[n + 1] = 0;
        for (int i = 1; i <= n; i++) {
            int j = n - i + 1;
            pre[i] = (pre[i - 1] * base + (ull)str[i]) % MOD;
            suf[j] = (suf[j + 1] * base + (ull)str[j]) % MOD;
        }
    }
    // 判断[l, r]子串是否是回文
    bool isPalindrome(int l, int r) {
        ull lef = hashL(l, r);
        ull rig = hashR(l, r);
        return lef == rig;
    }
} my_str;
```

## 9.3 KMP算法

```cpp
vector<int> GetPre(string &s){
    int n = s.length();
    vector<int> nxt(n, 0);
    int j = 0;
    for(int i = 1; i < n; i++){
        j = nxt[i - 1];
        while(j > 0 && s[i] != s[j]){
            j = nxt[j - 1];//上一项的前后缀刚好不等处
        }
        if(s[i] == s[j]) j++;
        nxt[i] = j;
    }
    return move(nxt);
}


vector<int> KMP(string &s, string &patt){
    int siz1 = s.size(), siz2 = patt.size();
    string temp = patt + "!" + s;
    vector<int> nxt = GetPre(temp), ans;
    
    for(int i = 2 * siz2 - 1; i < nxt.size(); i++){
        if(nxt[i] == siz2){
            ans.push_back(i - 2 * siz2);
        }
    }
    return move(ans);
}
```

## 9.4 Manacher算法

计算字符串的最长回文子串长度。线性dp，从对称中心向两边扩展。

```cpp
int manacher(string &s) {
    string t = "^ ";
    for (auto c : s) t += c, t += " ";
    t += "!";
    int n = (int)t.size();
    vector<int> p(n, 0);
    int c = 0, r = 0; // c为当前中心，r为当前中心的最右边界
    for (int i = 1; i < n - 1; ++i) {
        if (i <= r) p[i] = min(r - i, p[2*c - i]);
        while (t[i - p[i] - 1] == t[i + p[i] + 1]) p[i]++;
        if (i + p[i] > r) r = i + p[i], c = i;
    }
    return *max_element(p.begin(), p.end());
}
```

# 10 动态规划

## 10.1 最长上升子序列 LIS

```cpp
// 返回最长单调上升子序列, O(nlogn)
vector<int> LIS(vector<int>& arr) {
    int len = arr.size() - 1;  // 1-base
    int ans = 0;
    vector<int> low(2);
    low[1] = arr[1], ans = 1;
    for (int i = 2; i <= len; i++) {
        if (arr[i] > low[ans]) {
            low.push_back(arr[i]);
            ans++;
        } else {
            int p = lower_bound(low.begin() + 1, low.end(), arr[i]) - low.begin();
            low[p] = arr[i];
        }
    }
    return move(low);
}
```

## 10.2 换根dp

```cpp
void link(int x, int y) { // 子树y的贡献并入结点x
    dp[x] += dp[y];
}
void cut(int x, int y) { // 结点x删除子树y的贡献
    dp[x] -= dp[y];
}
void dfs1(int x, int fa, vector<vector<int>> &g) { // 统计子树对根的贡献
    dp[x] = 1;
    for (y : g[x]) {
        if (y == fa) continue;
        dfs1(y, x, g);
        link(x, y);
    }
}
void dfs2(int x, int fa, vector<vector<int>> &g) {
    ans[x] = dp[x]; // 进入结点时，统计答案
    for (int y : g[x]) {
        if (y == fa) continue;
        cut(x, y), link(y, x); // 换根
        dfs2(y, x, g);
        cut(y, x), link(x, y); // 回溯
    }
}
```

## 10.3 SOSdp

$O(n\cdot 2^n)$ 处理子集，超集的和。

```cpp
void sum_over_subsets_dp() {
    // 1. 初始化 DP 数组
    // F[mask] 的初始值就是 A[mask] 本身，因为 mask 是自身的子集
    for (int mask = 0; mask < (1 << N); ++mask) {
        F[mask] = A[mask];
    }

    // 2. 按位进行 DP
    // 外层循环遍历每一位 (维度)
    for (int i = 0; i < N; ++i) {
        // 内层循环遍历所有 mask
        for (int mask = 0; mask < (1 << N); ++mask) {
            // 如果 mask 的第 i 位是 1
            // 那么 F[mask] 的值需要加上 F[mask ^ (1 << i)] 的贡献
            // mask ^ (1 << i) 是 mask 去掉第 i 位的结果，是 mask 的一个子集
            if (mask & (1 << i)) {
                F[mask] += F[mask ^ (1 << i)];
            }
        }
    }
}

void sum_over_supersets_dp() {
    // 1. 初始化
    for (int mask = 0; mask < (1 << N); ++mask) {
        F[mask] = A[mask];
    }

    // 2. 按位进行 DP
    for (int i = 0; i < N; ++i) {
        // 内层循环 mask 需要从大到小
        for (int mask = (1 << N) - 1; mask >= 0; --mask) {
            // 如果 mask 的第 i 位是 0
            // 那么 F[mask] 的值需要加上 F[mask | (1 << i)] 的贡献
            // mask | (1 << i) 是 mask 加上第 i 位的结果，是 mask 的一个超集
            if (!(mask & (1 << i))) {
                F[mask] += F[mask | (1 << i)];
            }
        }
    }
}
```

# 11 杂项

## 11.1 快读

```cpp
char buf[1 << 20], *p1, *p2;
char gc() {
    return p1 == p2 ? p2 = buf + fread(p1 = buf, 1, 1 << 20, stdin), (p1 == p2) ? EOF : *p1++ : *p1++;
}
inline int read(int f = 1, char c = gc(), int x = 0) {
    while (c < '0' || c > '9') f = (c == '-') ? -1 : 1, c = gc();
    while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = gc();
    return f * x;
}
```

## 11.2 调试

```cpp
void pv(const T& v) {
    if constexpr (is_same_v<T, vector<int>>) {
        for (auto i : v) cout << i << " ";
    } else {
        cerr << v;
    }
}
#define debug(v) cerr << #v << " = ", pv(v), cerr << endl
```

## 11.3 __int128

```cpp
#include <bits/stdc++.h>
using namespace std;

using i128 = __int128;

ostream& operator<<(ostream& os, i128 x) {
    if (x < 0) {
        os << "-";
        x = -x;
    }
    if (x == 0) return os << 0;
    string s;
    while (x) {
        s += char('0' + x % 10);
        x /= 10;
    }
    reverse(s.begin(), s.end());
    return os << s;
}

istream& operator>>(istream& is, i128& x) {
    string s; is >> s;
    x = 0;
    for (auto c : s) {
        x = x * 10 + c - '0';
    }
    return is;
}

i128 sqrti128(i128 x) {
    i128 lo = 0, hi = 1e18;
    while (lo < hi) {
        i128 mid = (lo + hi + 1) / 2;
        if (mid * mid <= x) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}
```

## 11.4 实数域二分

```cpp
for (int t = 1; t <= 100; t++) {
    ld mid = (l + r) / 2;
    if (judge(mid)) r = mid;
    else l = mid;
}
cout << l << endl;
```

## 11.5 整数域三分

这段代码求的是**极小值**，极大值需要改变不等号方向。

```cpp
while (l < r) {
    int mid = (l + r) / 2;
    if (check(mid) <= check(mid + 1)) r = mid; // <========= 注意不等号方向
    else l = mid + 1;
}
cout << check(l) << endl;
```

## 11.6 实数域三分

```cpp
ld l = -1E9, r = 1E9;
for (int t = 1; t <= 100; t++) {
    ld mid1 = (l * 2 + r) / 3;
    ld mid2 = (l + r * 2) / 3;
    if (judge(mid1) < judge(mid2)) {
        r = mid2;
    } else {
        l = mid1;
    }
}
cout << l << endl;
```

## 11.7 pb_ds

```cpp
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;

// ordered_set: 支持 order_of_key / find_by_order
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

int main() {
    // 普通有序集合
    ordered_set<int> os;
    os.insert(5);
    os.insert(1);
    os.insert(7);
    os.insert(3);
    // find_by_order(k): 返回第 k (0-based) 小元素的迭代器
    cout << "2nd: " << *os.find_by_order(2) << "\n"; // 5

    // order_of_key(x): 返回集合中严格小于 x 的元素个数
    cout << "count < 5: " << os.order_of_key(5) << "\n"; // 2 (1,3)

    // 删除
    os.erase(3);
    cout << "after erase 3, count < 5: " << os.order_of_key(5) << "\n"; // 1
}
```

## 11.8 日期换算（基姆拉尔森公式）

已知年月日，求星期数。

```c++
int week(int y,int m,int d){
    if(m<=2)m+=12,y--;
    return (d+2*m+3*(m+1)/5+y+y/4-y/100+y/400)%7+1;
}
```

## 11.9 自适应辛普森法（Simpson）

二维平面上的曲线长度
$$
L = ∫_a^b \sqrt{1 + (f'(x))^2} dx
$$

$$
L = ∫_{t1}^{t2} \sqrt{(dx/dt)^2 + (dy/dt)^2} dt
$$

```cpp
const int MAX_DEPTH = 60;

double simpson(double l, double r) {
    return (f(l) + 4.0 * f((l + r) / 2.0) + f(r)) * (r - l) / 6.0;
}

double integral_rec(double l, double r, double eps, double st, int depth) {
    double mid = (l + r) / 2.0;
    double sl = simpson(l, mid), sr = simpson(mid, r);
    double I = sl + sr;
    double err = fabs(I - st) / 15.0; // 估计的误差

    // 复合绝对/相对容差
    double tol = max(eps, eps * fabs(I)); // 或：max(abs_tol, rel_tol * fabs(I))

    if (err <= tol || depth >= MAX_DEPTH) {
        return I + (I - st) / 15.0; // Richardson 修正
    }
    return integral_rec(l, mid, eps / 2.0, sl, depth + 1)
         + integral_rec(mid, r, eps / 2.0, sr, depth + 1);
}

double integral(double l, double r, double EPS) {
    double s = simpson(l, r);
    return integral_rec(l, r, EPS, s, 0);
}
```

## 11.10 整数除法

```cpp
int floordiv(int x, int y) {
    int q = x / y;
    if (x % y and (x > 0) != (y > 0)) q--;
    return q;
}
int ceildiv(int x, int y) {
    int q = x / y;
    if (x % y and (x > 0) == (y > 0)) q++;
    return q;
}
```

