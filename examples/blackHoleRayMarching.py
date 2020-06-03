import taichi as ti
import numpy as np

# Shadertoy: BlackHole-RayMarching, reference => https://www.shadertoy.com/view/3slcWr

ti.init(debug=False, arch=ti.gpu)

PI = 3.1415926
WIDTH, HEIGHT = 640, 320
pixels = ti.Vector(4, dt=ti.f32, shape=(WIDTH, HEIGHT))

@ti.func
def rotate(a):
    s = ti.sin(a)
    c = ti.cos(a)
    return ti.Matrix([[c, -s], [s, c]])

@ti.func
def fract(x):
    return x - ti.floor(x)

@ti.func
def hash(p):
    p = fract(p * 0.3183099 + 0.1)
    p *= 17.0
    return fract(p[0] * p[1] * p[2] * (p[0] + p[1] + p[2]))

@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a

@ti.func
def clamp(x, a_min, a_max):
    return min(max(x, a_min), a_max)

@ti.func
def sphere(s):
    return ti.Vector([s[0], s[1], s[2]]).norm() - s[3]

@ti.func
def noise(x):
    i = ti.floor(x)
    f = fract(x)
    u = f * f * (3.0 - 2.0 * f)
    
    v1 = mix(hash(i + ti.Vector([0.0, 0.0, 0.0])), 
             hash(i + ti.Vector([1.0, 0.0, 0.0])), u[0])
    v2 = mix(hash(i + ti.Vector([0.0, 1.0, 0.0])), 
             hash(i + ti.Vector([1.0, 1.0, 0.0])), u[0])
    v12 = mix(v1, v2, u[1])
    
    v3 = mix(hash(i + ti.Vector([0.0, 0.0, 1.0])), 
             hash(i + ti.Vector([1.0, 0.0, 1.0])), u[0])
    v4 = mix(hash(i + ti.Vector([0.0, 1.0, 1.0])), 
             hash(i + ti.Vector([1.0, 1.0, 1.0])), u[0])
    v34 = mix(v3, v4, u[1])

    return mix(v12, v34, u[2])

@ti.func
def getGlow(minPDist):
    
    mainGlow = minPDist * 1.2
    mainGlow = pow(mainGlow, 32.0)
    mainGlow = clamp(mainGlow, 0.0, 1.0)
    
    outerGlow = minPDist * 0.4
    outerGlow = pow(outerGlow, 2.0)
    outerGlow = clamp(outerGlow, 0.0, 1.0)

    return ti.Vector([10.0, 5.0, 3.0, min(mainGlow + outerGlow, 1.0)])

@ti.func
def getDist(p):
    diskPos = -1.0 * p
    diskDist = sphere(ti.Vector([diskPos[0], diskPos[1], diskPos[2], 5.0]))
    diskDist = max(diskDist, diskPos[1] - 0.01)
    diskDist = max(diskDist, -1.0 * diskPos[1] - 0.01)
    diskDist = max(diskDist, -1.0 * sphere(ti.Vector([-1.0 * p[0], -1.0 * p[1], -1.0 * p[2], 1.5]) * 10.0))
    if diskDist < 2.0:
        c = ti.Vector([diskPos.norm(), diskPos[1], ti.atan2(diskPos[2] + 1.0, diskPos[0] + 1.0) * 0.5])
        c *= 10.0
        diskDist += noise(c) * 0.4
        diskDist += noise(c * 2.5) * 0.2
    return diskDist

@ti.func
def raymarch(ro, rd):
    p = ro
    glow = 0.0
    for _ in range(700):
        dS = getDist(p)
        glow = max(glow, 1.0 / (dS + 1.0))
        bdir = -1.0 * p.normalized()
        bdist = p.norm()
        dS = min(dS, bdist) * 0.04
        if dS > 30.0:
            break
        if bdist < 1.0:
            break
        bdist = pow(bdist + 1.0, 2.0)
        bdist = dS * 1.0 / bdist
        rd = mix(rd, bdir, bdist)
        p += rd * max(dS, 0.01)
    gcol = getGlow(glow)
    c_rgb = mix(ti.Vector([0.0, 0.0, 0.0]), ti.Vector([gcol[0], gcol[1], gcol[2]]), gcol[3])
    return ti.Vector([c_rgb[0], c_rgb[1], c_rgb[2], 1.0])

@ti.kernel
def render(t: ti.f32):

    for i, j in pixels:
        uv = (ti.Vector([i, j]) - 0.5 * ti.Vector([WIDTH, HEIGHT])) / HEIGHT
        ro = ti.Vector([0.0, ti.cos(t * 0.5) * 10.0, ti.sin(t * 0.5) * 10.0])
        rd = ti.Vector([uv[0], uv[1], 1.0]).normalized()

        # rotate
        rmat = rotate(0.5 * (t + PI))
        new_rdx = rmat @ ti.Vector([rd[1], rd[2]])
        rd[1] = new_rdx[0]
        rd[2] = new_rdx[1]

        # color
        pixels[i, j] = raymarch(ro, rd)


gui = ti.GUI("BlackHole-RayMarching", res=(WIDTH, HEIGHT))

def main():
    for ts in range(1000000):
        render(ts * 0.03)
        gui.set_image(pixels.to_numpy())
        gui.show()


if __name__ == "__main__":
    
    main()
