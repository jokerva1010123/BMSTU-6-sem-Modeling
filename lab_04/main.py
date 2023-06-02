import math
import matplotlib.pyplot as plt

class Model:
    a2 = 2.049
    b2 = 0.563e-3
    c2 = 0.528e5
    m2 = 1
    tau = 4e-6
    Fmax = 5000
    tmax = 150e-6

    def __init__(self, tau: float, np: float, r0: float, r: float, zN: float, t0: float, sigma: float, f0: float, alpha: float, h: float):
        self.tau = tau
        self.Np = np
        self.r0 = r0
        self.R = r
        self.zN = zN
        self.T0 = t0
        self.sigma = sigma
        self.alpha = alpha
        self.h = h
        self.z0 = r0 / self.R


    def Labmda(self, T: float):
        lambdaArr = [1.36e-2, 1.63e-2, 1.81e-2, 1.98e-2, 2.50e-2, 2.74e-2]
        t = [300, 500, 800, 1100, 2000, 2400]
        return self.interpolate(T, t, lambdaArr)

    def K(self, T: float):
        k = [2.0e-2, 5.0e-2, 7.8e-2, 1.0e-1, 1.3e-1, 2.0e-1]
        t = [293, 1278, 1528, 1677, 2000, 2400]
        return self.interpolate(T, t, k)

    def F0(self, t: float):
        # if (t < 100):
        #     return 100
        # else:
        #     return 0
        f0 = self.Fmax / self.tmax * t * math.exp(-(t / self.tmax - 1))
        return f0
        
    def P(self, T: float):
        return 4 * self.Np * self.Np * self.sigma * self.K(T) * math.pow(T, 3)

    def F(self, T: float):
        return 4 * self.Np * self.Np * self.sigma * self.K(T) * math.pow(self.T0, 4)

    def V(self, zn_1: float, zn: float, zn1: float):
        return (math.pow((zn1 + zn) / 2, 2) - math.pow((zn + zn_1) / 2, 2)) / 2

    def Ct(self, yn: float):
        return self.a2 + self.b2 * math.pow(yn, self.m2) - self.c2 / (yn * yn)

    def A(self, yn_1: float, yn: float, zn_1: float, zn: float):
        ln_12 = (self.Labmda(yn_1) + self.Labmda(yn)) / 2
        zn_12 = (zn_1 + zn) / 2
        return zn_12 * ln_12 / (self.R * self.R * self.h) * self.tau

    def B(self, yn_1: float, yn: float, yn1: float, zn_1: float, zn: float, zn1: float):
        return self.A(yn_1, yn, zn_1, zn) + self.C(yn, yn1, zn, zn1) \
            + self.P(yn) * self.V(zn_1, zn, zn1) * self.tau + zn * self.Ct(yn) * self.h

    def C(self, yn: float, yn1: float, zn: float, zn1: float):
        ln12 = (self.Labmda(yn1) + self.Labmda(yn)) / 2
        zn12 = (zn1 + zn) / 2
        return zn12 * ln12 / (self.R * self.R * self.h) * self.tau

    def D(self, yn: float, zn_1: float, zn: float, zn1: float, ym: float):
        return self.F(yn) * self.V(zn_1, zn, zn1) * self.tau + zn * self.Ct(yn) * self.h * ym
        

    def interpolate(self, xval: float, xs, ys): # xs, ys = [float]
        i = 0
        while (i < len(xs) and xval < xs[i]):
            i += 1
        if (i >= len(xs) - 1):
            i = len(xs) - 2
        yval = ys[i] + (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) * (xval - xs[i])
        return yval
    
    # a, b, c, d - [float]
    def run_method(self, a, b, c, d, m0: float, k0: float, mn: float, kn: float, p0: float, pn: float): # -> [float]
        y = [0] * len(b)
        xi = [0] * (len(b) - 1)
        eta = [0] * (len(b) - 1)
        xi[0] = -k0 / m0
        eta[0] = p0 / m0
        for i in range(1, len(xi)):
            tmp = b[i] - a[i] * xi[i - 1]
            xi[i] = c[i] / tmp
            eta[i] = (a[i] * eta[i - 1] + d[i]) / tmp

        y[len(y) - 1] = 500 # (pn - mn * eta[len(eta) - 1]) / (kn + mn * xi[len(xi) - 1]) # выраженное y n-ое 
        for i in range(len(y) - 1, 0, -1):
            y[i - 1] = xi[i - 1] * y[i] + eta[i - 1] # та самая формула
        return y
    
    def P0(self, z0: float, z1: float, y0: float, y1: float, t: float, ym0: float, ym1: float):
        z12 = (z0 + z1) / 2
        f12 = (self.F(y0) + self.F(y1)) / 2
        c12 = (self.Ct(y0) + self.Ct(y1)) / 2
        ym12 = (ym0 + ym1) / 2
        return (z0 * self.Ct(y0) * ym0 + z12 * c12 * ym12) * self.h / 2  \
            + z0 / self.R * self.F0(t) * self.tau + self.h / 4 * (self.F(y0) * z0 + f12 * z12) * self.tau
        # return F0;

    def K0(self, z0: float, z1: float, y0: float, y1: float):
        z12 = (z0 + z1) / 2
        x12 = (self.Labmda(y0) + self.Labmda(y1)) / 2
        p12 = (self.P(y0) + self.P(y1)) / 2
        c12 = (self.Ct(y0) + self.Ct(y1)) / 2
        return self.h / 4 * c12 * z12 - z12 * x12 * self.tau / (self.R * self.R * self.h) + self.h / 8 * p12 * z12 * self.tau
        # return -x12 / (R * h);

    def M0(self, z0: float, z1: float, y0: float, y1: float):
        z12 = (z0 + z1) / 2
        x12 = (self.Labmda(y0) + self.Labmda(y1)) / 2
        p12 = (self.P(y0) + self.P(y1)) / 2
        c12 = (self.Ct(y0) + self.Ct(y1)) / 2
        return self.h / 2 * (c12 * z12 / 2 + z0 * self.Ct(y0)) + z12 * x12 * self.tau / (self.R * self.R * self.h) \
                + self.h * self.tau / 4 * (p12 * z12 / 2 + self.P(y0) * z0)
        # return x12 / (R * h);

    def PN(self, zN_1: float, zN: float, yN_1: float, yN: float, ymN: float, ymN_1: float):
        zN_12 = (zN_1 + zN) / 2
        fN_12 = (self.F(yN_1) + self.F(yN)) / 2
        cN_12 = (self.Ct(yN) + self.Ct(yN_1)) / 2
        return (zN_12 * cN_12 + zN * self.Ct(yN)) * self.h / 2 + self.alpha * self.T0 * zN * self.tau / self.R \
                + self.h / 4 * (self.F(yN) * zN + fN_12 * zN_12) * self.tau
        # return alpha * T0;

    def KN(self, zN_1: float, zN: float, yN_1: float, yN: float):
        zN_12 = (zN + zN_1) / 2
        xN_12 = (self.Labmda(yN_1) + self.Labmda(yN)) / 2
        pN_12 = (self.P(yN) + self.P(yN_1)) / 2
        cN_12 = (self.Ct(yN) + self.Ct(yN_1)) / 2
        return self.h / 2 * (zN_12 * cN_12 / 2 + zN * self.Ct(yN)) + zN_12 * xN_12 * self.tau / (self.R * self.R * self.h) \
                + self.alpha * zN * self.tau / self.R + self.h * self.tau / 4 * (pN_12 * zN_12 / 2 + self.P(yN) * zN)
        # return -xN_12 / (R * h) + alpha;

    def MN(self, zN_1: float, zN: float, yN_1: float, yN: float):
        zN_12 = (zN + zN_1) / 2
        xN_12 = (self.Labmda(yN) + self.Labmda(yN_1)) / 2
        pN_12 = (self.P(yN) + self.P(yN_1)) / 2
        cN_12 = (self.Ct(yN) + self.Ct(yN_1)) / 2
        return self.h / 4 * zN_12 * cN_12 - zN_12 * xN_12 * self.tau / (self.R * self.R * self.h) + self.h / 8 * pN_12 * zN_12 * self.tau
        # return xN_12 / (R * h);
        
    # -> [float]
    # z, ym - [float]
    def next_iter(self, z, ym, t: float):
        n = len(z)
        y = [0] * n
        y = ym.copy()
        erry = [0] * n
        its = 0
        # метод простых итераций
        while True:
            # вычисление коэффициентов
            a = [0] * n
            b = [0] * n
            c = [0] * n
            d = [0] * n
            for i in range(1, n - 1):
                a[i] = self.A(y[i - 1], y[i], z[i - 1], z[i])
                b[i] = self.B(y[i - 1], y[i], y[i + 1], z[i - 1], z[i], z[i + 1])
                c[i] = self.C(y[i], y[i + 1], z[i], z[i + 1])
                d[i] = self.D(y[i], z[i - 1], z[i], z[i + 1], ym[i])
            k0 = self.K0(z[0], z[1], y[0], y[1])
            m0 = self.M0(z[0], z[1], y[0], y[1])
            p0 = self.P0(z[0], z[1], y[0], y[1], t, ym[0], ym[1])
            kn = self.KN(z[n - 2], z[n - 1], y[n - 2], y[n - 1])
            mn = self.MN(z[n - 2], z[n - 1], y[n - 2], y[n - 1])
            pn = self.PN(z[n - 2], z[n - 1], y[n - 2], y[n - 1], ym[n - 1], ym[n - 2])
            # метод прогонки
            newy = self.run_method(a, b, c, d, m0, k0, mn, kn, p0, pn)

            # высчитваем ошибку по текущей итерации
            for i in range(0, n):
                erry[i] = math.fabs((newy[i] - y[i]) / newy[i])
            y = newy
            maxerr = max(erry)
            its += 1
            if not (maxerr > 1e-3 and its < 50):
                break
        print(its)
        return y

x = []
y = []

mod = Model(tau=3e-6, np=1.4, r0=0.35, r=0.5, zN=1, t0=300, sigma=5.668e-12, f0=100.0, alpha=0.05, h=2e-3)
n = int(((mod.zN - mod.z0) / mod.h / mod.R)) + 1 # кол-во шагов
z = [0] * n
y = [0] * n
x = [0] * n
t = [0] * n
ys = []

for i in range(0, n):
    z[i] = mod.z0 + i * mod.h / mod.R
    x[i] = mod.z0 + i * mod.h * mod.R
    y[i] = mod.T0

i = 0
while(i < 2000e-6):
    tmp = [0] * n
    tmp = y.copy()
    ys.append(tmp)
    y = mod.next_iter(z, y, i) 
    i += mod.tau

#plt.figure(figsize=(30, 10))


for j in range(0, 15):
    a = []
    b = []
    for i in range(0, n):
        a.append(x[i])
        b.append(ys[(len(ys) - 1) // 15 * j][i])
    #plt.plot(a, b) # завимость температуры от отдаления от внутренней стенки (разные графики показывают это распределение в разный момент времени)

newX = []
newY = []
for i in range(0, len(ys)):
    newX.append(i * mod.tau)
    newY.append(ys[i][0])

plt.plot(newX, newY) # зависимость температуры внутренней стенки труюдки от времени

plt.legend()
plt.show()