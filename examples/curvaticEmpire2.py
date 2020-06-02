import taichi as ti
import numpy as np

# Shadertoy: curvaticEmpire2, reference => https://www.shadertoy.com/view/lstSzj

ti.init(debug=False, arch=ti.gpu)

m, n = 640, 320
pixels = ti.Vector(3, dt=ti.f32, shape=(m, n))

@ti.kernel
def render(t: ti.f32):

    # camera & camera tx
    ro = ti.Vector([ti.sin(t * 0.16), 0.0, ti.cos(t * 0.1)])
    ta = ro + ti.Vector([ti.sin(t * 0.15), ti.sin(t * 0.18), ti.cos(t * 0.24)])

    cw = (ta - ro).normalized()
    cp = ti.Vector([0.0, 1.0, 0.0])
    cu = cp.cross(cw).normalized()

    for i, j in pixels:

        # coords-trans
        coords = -1.0 + 2.0 * ti.Vector([i / m, j / n])
        coords[0] *= (m / n)

        rd = (coords[0] * cu + coords[1] * cp + 2.0 * cw).normalized()

        v = ti.Vector([0.0, 0.0, 0.0])
        for k1 in range(50):
            s1 = (k1 + 1) * 0.1
            p = ro + rd * s1
            for k2 in range(8):
                s2 = 0.1 + 0.12 * k2
                p = abs(p) / (p + ti.sin(t * 0.1) * 0.21).dot(p) - 0.85
                a = p.norm() * 0.12
                v += ti.Vector([a * s2, a * s2 ** 2, a * s2 ** 3])
        pixels[i, j] = v * 0.01

gui = ti.GUI("curvaticEmpire2", res=(m, n))

for ts in range(1000000):
    render(ts * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
