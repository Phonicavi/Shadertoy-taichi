import taichi as ti
import numpy as np

# Shadertoy: curvaticEmpire2, reference => https://www.shadertoy.com/view/Ms2SD1

ti.init(debug=False, arch=ti.gpu)

WIDTH, HEIGHT = 640, 320
NUM_STEPS = 8
PI = 3.141592
EPSILON = 1e-3
EPSILON_NRM = (0.1 / WIDTH)

# sea
ITER_GEOMETRY = 3
ITER_FRAGMENT = 5
SEA_HEIGHT = 0.6
SEA_CHOPPY = 4.0
SEA_SPEED = 0.8
SEA_FREQ = 0.16
SEA_BASE = ti.Vector([0.0, 0.09, 0.18])
SEA_WATER_COLOR = ti.Vector([0.8, 0.9, 0.6]) * 0.6

@ti.func
def sea_time(timef32):
    return 1.0 + timef32 * SEA_SPEED

pixels = ti.Vector(3, dt=ti.f32, shape=(WIDTH, HEIGHT))
octave_m_v1 = ti.Vector([1.6, 1.2])
octave_m_v2 = ti.Vector([-1.2, 1.6])

# device
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        mouse_data = np.array([0] * 4, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mgp = gui.get_cursor_pos()
            mxy = np.array([mgp[0] * WIDTH, mgp[1] * HEIGHT], dtype=np.float32)
            if self.prev_mouse is None:
                self.prev_mouse = mxy
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
        return mouse_data


# math
@ti.func
def fromEuler(ang):
    a1 = ti.Vector([ti.sin(ang[0]), ti.cos(ang[0])])
    a2 = ti.Vector([ti.sin(ang[1]), ti.cos(ang[1])])
    a3 = ti.Vector([ti.sin(ang[2]), ti.cos(ang[2])])
    
    m = ti.Matrix([
        [a1[1] * a3[1] + a1[0] * a2[0] * a3[0], a1[1] * a2[0] * a3[0] + a3[1] * a1[0], -1.0 * a2[1] * a3[0]],
        [-1.0 * a2[1] * a1[0], a1[1] * a2[1], a2[0]],
        [a3[1] * a1[0] * a2[0] + a1[1] * a3[0], a1[0] * a3[0] - a1[1] * a3[1] * a2[0], a2[1] * a3[1]]
    ])
    return m

@ti.func
def fract(x):
    return x - ti.floor(x)

@ti.func
def hash(p):
    h = p.dot(ti.Vector([127.1, 311.7]))
    return fract(ti.sin(h) * 43758.5453123)

@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a

@ti.func
def reflect(i, n):
    return i - 2.0 * n.dot(i) * n

@ti.func
def clamp(x, a_min, a_max):
    return min(max(x, a_min), a_max)

@ti.func
def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def noise(p):
    i = ti.floor(p)
    f = fract(p)
    u = f * f * (3.0 - 2.0 * f)
    v1 = mix(hash(i + ti.Vector([0.0, 0.0])), hash(i + ti.Vector([1.0, 0.0])), u[0])
    v2 = mix(hash(i + ti.Vector([0.0, 1.0])), hash(i + ti.Vector([1.0, 1.0])), u[0])
    v3 = mix(v1, v2, u[1])
    return -1.0 + 2.0 * v3

# lighting
@ti.func
def diffuse(n, l, p):
    return pow(n.dot(l) * 0.4 + 0.6, p)

@ti.func
def specular(n, l, e, s):
    nrm = (s + 8.0) / (PI * 3.0)
    return pow(max(reflect(e, n).dot(l), 0.0), s) * nrm

# sky
@ti.func
def getSkyColor(e):
    e[1] = (max(e[1], 0.0) * 0.8 + 0.2) * 0.8
    s = 1.0 - e[1]
    return ti.Vector([s ** 2, s, 0.6 + s * 0.4]) * 1.1

# sea
@ti.func
def sea_octave(uv, choppy):
    uv += noise(uv)
    wv = 1.0 - abs(ti.sin(uv))
    swv = abs(ti.cos(uv))
    wv = mix(wv, swv, wv)
    return pow(1.0 - pow(wv[0] * wv[1], 0.65), choppy)

@ti.func
def map(p, timef32):
    freq = SEA_FREQ
    amp = SEA_HEIGHT
    choppy = SEA_CHOPPY
    uv = ti.Vector([p[0] * 0.75, p[2]])

    d = 0.0
    h = 0.0
    for _ in range(ITER_GEOMETRY):
        d = sea_octave((uv + sea_time(timef32)) * freq, choppy)
        d += sea_octave((uv - sea_time(timef32)) * freq, choppy)
        h += d * amp
        # uv *= octave_m
        uv = ti.Vector([uv.dot(octave_m_v1), uv.dot(octave_m_v2)])
        freq *= 1.9
        amp *= 0.22
        choppy = mix(choppy, 1.0, 0.2)
    return p[1] - h

@ti.func
def map_detailed(p, timef32):
    freq = SEA_FREQ
    amp = SEA_HEIGHT
    choppy = SEA_CHOPPY
    uv = ti.Vector([p[0] * 0.75, p[2]])

    d = 0.0
    h = 0.0
    for _ in range(ITER_FRAGMENT):
        d = sea_octave((uv + sea_time(timef32)) * freq, choppy)
        d += sea_octave((uv - sea_time(timef32)) * freq, choppy)
        h += d * amp
        # uv *= octave_m
        uv = ti.Vector([uv.dot(octave_m_v1), uv.dot(octave_m_v2)])
        freq *= 1.9
        amp *= 0.22
        choppy = mix(choppy, 1.0, 0.2)
    return p[1] - h

@ti.func
def getSeaColor(p, n, l, eye, dist):
    fresnel = clamp(1.0 - n.dot(-1.0 * eye), 0.0, 1.0)
    fresnel = pow(fresnel, 3.0) * 0.5

    reflected = getSkyColor(reflect(eye, n))
    refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12

    color = mix(refracted, reflected, fresnel)

    atten = max(1.0 - dist.dot(dist) * 0.001, 0.0)
    color += SEA_WATER_COLOR * (p[1] - SEA_HEIGHT) * 0.18 * atten

    sp = specular(n, l, eye, 60.0)
    color += ti.Vector([sp, sp, sp])

    return color

# tracing
@ti.func
def getNormal(p, eps, timef32):
    ny = map_detailed(p, timef32)
    nx = map_detailed(ti.Vector([p[0] + eps, p[1], p[2]]), timef32) - ny
    nz = map_detailed(ti.Vector([p[0], p[1], p[2] + eps]), timef32) - ny
    ny = eps
    return ti.Vector([nx, ny, nz]).normalized()

@ti.func
def heightMapTracing(ori, dir, p, timef32):
    tm, tx = 0.0, 1000.0
    hx = map(ori + dir * tx, timef32)
    if hx <= 0.0:
        hm = map(ori + dir * tm, timef32)
        tmid = 0.0
        for _ in range(NUM_STEPS):
            tmid = mix(tm, tx, hm / (hm - hx))
            p = ori + dir * tmid
            hmid = map(p, timef32)
            if hmid < 0.0:
                tx = tmid
                hx = hmid
            else:
                tm = tmid
                hm = hmid
    return p

@ti.func
def getPixel(coord, timef32):
    uv = -1.0 + 2.0 * ti.Vector([coord[0] / WIDTH, coord[1] / HEIGHT])
    uv[0] *= (WIDTH / HEIGHT)

    # ray
    ang = ti.Vector([ti.sin(timef32 * 3.0) * 0.1, ti.sin(timef32) * 0.2 + 0.3, timef32])
    ori = ti.Vector([0.0, 3.5, timef32 * 5.0])
    dir = ti.Vector([uv[0], uv[1], -2.0]).normalized()
    dir[2] += uv.norm() * 0.14
    dir_r = dir.normalized()
    me = fromEuler(ang)
    dir = ti.Vector([
        dir_r.dot(ti.Vector([me[0, 0], me[0, 1], me[0, 2]])),
        dir_r.dot(ti.Vector([me[1, 0], me[1, 1], me[1, 2]])),
        dir_r.dot(ti.Vector([me[2, 0], me[2, 1], me[2, 2]]))
    ])

    # tracing
    p = ti.Vector([0.0, 0.0, 0.0])
    p = heightMapTracing(ori, dir, p, timef32)
    dist = p - ori
    n = getNormal(p, dist.dot(dist) * EPSILON_NRM, timef32)
    light = ti.Vector([0.0, 1.0, 0.8]).normalized()

    # color
    return mix(
        getSkyColor(dir), 
        getSeaColor(p, n, light, dir, dist), 
        pow(smoothstep(0.0, -0.02, dir[1]), 0.2)
    )

@ti.kernel
def render(t: ti.f32, mouse: ti.ext_arr()):
    
    timef32 = t * 0.3 + mouse[2] * 0.01
    for i, j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                uv = ti.Vector([i + dx / 3.0, j + dy / 3.0])
                color += getPixel(uv, timef32)
        color /= 9.0
        pixels[i, j] = pow(color, 0.65)


def main():
    gui = ti.GUI("seascape", res=(WIDTH, HEIGHT))
    md_gen = MouseDataGen()
    paused = False
    ts = 0
    while True:
        while gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                exit(0)
            elif e.key == 'p':
                paused = not paused

        if not paused:
            mouse_data = md_gen(gui)
            render(ts * 0.03, mouse_data)
            ts += 1

        img = pixels.to_numpy()
        gui.set_image(img)
        gui.show()


if __name__ == "__main__":
    
    main()
