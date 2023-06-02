import math
import matplotlib.pyplot as plt

class Model:
    Np = 1.4
    r0 = 0.35
    R = 0.5
    z0 = r0 / R
    zN = 1
    T0 = 300
    sigma = 5.668e-12
    f = 100
    alpha = 0.05
    h = 1e-4

    def F(self, T):
        return 4 * math.pow(self.Np, 2) * self.sigma * self.K(T) * math.pow(self.T0, 4)

    def P(self, T):
        return -4 * self.Np * self.Np * self.sigma * self.K(T) * math.pow(T, 3)

    def Interpolation(self, x0, x1, y1):
        i = 0
        while (i < len(x1) and x0 < x1[i]):
            i += 1
        if (i == len(x1) or i == 0):
            y0 = y1[0] + (y1[len(y1) - 1] - y1[0]) / (x1[len(x1) - 1] - x1[0]) * x0
            return y0
        y0 = y1[i - 1] + (y1[i] - y1[i - 1]) / (x1[i] - x1[i - 1]) * x0
        return y0

    def Labmda(self, T):
        lambdaArr = [1.36e-2, 1.63e-2, 1.81e-2, 1.98e-2, 2.50e-2, 2.74e-2]
        t = [300, 500, 800, 1100, 2000, 2400]
        return self.Interpolation(T, t, lambdaArr)
    
    def K(self, T):
        k = [2.0e-2, 5.0e-2, 7.8e-2, 1.0e-1, 1.3e-1, 2.0e-1]
        t = [293, 1278, 1528, 1677, 2000, 2400]
        return self.Interpolation(T, t, k)
    
    def V(self, zn_1, zn, zn1):
        return (math.pow((zn1  + zn) / 2, 2) - math.pow((zn + zn_1) / 2, 2)) / 2

    def A(self, yn_1, yn, zn_1, zn):
        return 1 / (self.R * self.R * self.h) * (zn_1 + zn) / 2 * (self.Labmda(yn_1) + self.Labmda(yn)) / 2
    
    def B(self, yn_1, yn, yn1, zn_1, zn, zn1):
        return self.A(yn_1, yn, zn_1, zn) + self.C(yn, yn1, zn, zn1) + self.P(yn) * self.V(zn_1, zn, zn1)

    def C(self, yn, yn1, zn, zn1):
        return 1 / (self.R * self.R * self.h) * (zn1 + zn) / 2 * (self.Labmda(yn1) + self.Labmda(yn)) / 2

    def D(self, yn, zn_1, zn, zn1):
        return self.F(yn) * self.V(zn_1, zn, zn1)
    
    def P0(self, z0, z1, y0, y1):
        z12 = (z0 + z1) / 2
        f12 = (self.F(y0) + self.F(y1)) / 2
        return z0 * self.f / self.R + self.h / 4 * (self.F(y0) * z0 + f12 * z12)

    def K0(self, z0, z1, y0, y1):
        z12 = (z0 + z1) / 2
        x12 = (self.Labmda(y0) + self.Labmda(y1)) / 2
        p12 = (self.P(y0) + self.P(y1)) / 2
        return -(z12 * x12) / (self.R * self.R * self.h) + self.h / 8 * p12 * z12

    def M0(self, z0, z1, y0, y1):
        z12 = (z0 + z1) / 2
        x12 = (self.Labmda(y0) + self.Labmda(y1)) / 2
        p12 = (self.P(y0) + self.P(y1)) / 2
        return (z12 * x12) / (self.R * self.R * self.h) + self.h / 8 * p12 * z12 + self.h / 4 * self.P(y0) * z0

    def PN(self, zN_1, zN, yN_1, yN):
        zN_12 = (zN_1 + zN) / 2
        fN_12 = (self.F(yN_1) + self.F(yN)) / 2
        return -self.alpha * self.T0 * zN / self.R + self.h / 4 * (self.F(yN) * zN + fN_12 * zN_12)
        
    def KN(self, zN_1, zN, yN_1, yN):
        zN_12 = (zN + zN_1) / 2
        xN_12 = (self.Labmda(yN_1) + self.Labmda(yN)) / 2
        pN_12 = (self.P(yN) + self.P(yN_1)) / 2
        return ((zN_12 * xN_12) / (self.R * self.R * self.h) + self.alpha * zN / self.R + self.h / 8 * pN_12 * zN_12 + self.h / 4 * self.P(yN) * zN)

    def MN(self, zN_1, zN, yN_1, yN):
        zN_12 = (zN + zN_1) / 2
        xN_12 = (self.Labmda(yN) + self.Labmda(yN_1)) / 2
        pN_12 = (self.P(yN) + self.P(yN_1)) / 2
        return -((zN_12 * xN_12) / (self.R * self.R * self.h) + self.h / 8 * pN_12 * zN_12)
    
    def run_inner(self, a, b, c, d, m0, k0, mn, kn, p0, pn):
        y = [0] * len(b)
        xi = [0] * (len(b) - 1)
        eta = [0] * (len(b) - 1)
        xi[0] = -k0 / m0
        eta[0] = p0 / m0
        for i in range(1, len(xi)):
            tmp = (b[i] - a[i] * xi[i - 1])
            xi[i] = c[i] / tmp
            eta[i] = (a[i] * eta[i - 1] + d[i]) / tmp
        y[len(y) - 1] = (pn - mn * eta[len(eta) - 1]) / (kn - mn * xi[len(xi) - 1])
        for i in range(len(y) - 1 , 0, -1):
            y[i - 1] = xi[i - 1] * y[i] + eta[i - 1]
        return y
    
    def run(self):
        n = int((self.zN - self.z0) / self.h)
        n += 1
        y = [0] * n
        z = [0] * n
        for i in range(n):
            z[i] = self.z0 + i * self.h
            y[i] = self.T0
        
        erry = [0] * n
        maxerr = 1
        its = 0
        while (maxerr > 1e-5 and its < 500):
            a = [0] * n
            b = [0] * n
            c = [0] * n
            d = [0] * n
            for i in range(1, n - 1):
                a[i] = self.A(y[i - 1], y[i],           z[i - 1], z[i])
                b[i] = self.B(y[i - 1], y[i], y[i + 1], z[i - 1], z[i], z[i + 1])
                c[i] = self.C(          y[i], y[i + 1],           z[i], z[i + 1])
                d[i] = self.D(          y[i],           z[i - 1], z[i], z[i + 1])
            k0 = self.K0(z[0], z[1], y[0], y[1])
            m0 = self.M0(z[0], z[1], y[0], y[1])
            p0 = self.P0(z[0], z[1], y[0], y[1])
            kn = self.KN(z[n - 2], z[n - 1], y[n - 2], y[n - 1])
            mn = self.MN(z[n - 2], z[n - 1], y[n - 2], y[n - 1])
            pn = self.PN(z[n - 2], z[n - 1], y[n - 2], y[n - 1])
            newy = self.run_inner(a, b, c, d, m0, k0, mn, kn, p0, pn)
                
            for i in range(n):
                erry[i] = math.fabs((newy[i] - y[i]) / newy[i])
            y = newy
            maxerr = max(erry)
            its += 1

        print(its)
        return (z, y)
    

m = Model()
x, y = m.run()

print(m.r0 * m.f)
print(m.R * m.alpha * (y[len(y) - 1] - m.T0))

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.legend()
plt.show()