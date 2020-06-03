from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.transform.rotation import Rotation as scirot
import matplotlib.pyplot as plt

import taichi as ti
import numpy as np

ti.init(debug=True, arch=ti.gpu)

RES = 512
K = ti.Matrix([[RES, 0.0, 0.5 * RES], [0.0, RES, 0.5 * RES], [0.0, 0.0, 1.0]])  # fov = atan2(1, 2), i.e. arctan(0.5)
PI = float(np.pi)
t_pushed = 3.0 * PI
t_farest = t_pushed + 2.0 * PI

nx, ny, nz = 30, 30, 30
pixels = ti.Vector(3, dt=ti.f32, shape=(RES, RES))
z_buff = ti.Vector(4, dt=ti.f32, shape=(RES, RES))  # z_buffer: [r,g,b,depth]
axis_angle = ti.Vector(3, dt=ti.f32, shape=(nx, ny, nz))  # axis:     [x,y,z]
aa_colormp = ti.Vector(4, dt=ti.f32, shape=(nx, ny, nz))  # cm: [r,g,b,depth]
aa_transfm = ti.Vector(2, dt=ti.i32, shape=(nx, ny, nz))  # px:         [x,y]


@ti.func
def colormap(aa):
    return aa / (2.0 * PI) + 0.5

@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a

@ti.func
def euler2rotmat(rx, ry, rz):
    # assume euler-order: 'rpy', roll(Z)->pitch(X)->yaw(Y)
    # i.e. X => pitch, Y => yaw, Z => roll
    cx, cy, cz = ti.cos(rx), ti.cos(ry), ti.cos(rz)
    sx, sy, sz = ti.sin(rx), ti.sin(ry), ti.sin(rz)

    m00 = cx * cz + sx * sy * sz
    m01 = cz * sx * sy - cx * sz
    m02 = cy * sx
    m10 = cy * sz
    m11 = cy * cz
    m12 = -sy
    m20 = cx * sy * sz - sx * cz
    m21 = sx * sz + cx * cz * sy
    m22 = cx * cy

    return ti.Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

@ti.func
def rotmat2quaternion(rotmat):
    m00, m01, m02 = rotmat[0, 0], rotmat[0, 1], rotmat[0, 2]
    m10, m11, m12 = rotmat[1, 0], rotmat[1, 1], rotmat[1, 2]
    m20, m21, m22 = rotmat[2, 0], rotmat[2, 1], rotmat[2, 2]

    tr = m00 + m11 + m22  # ti.tr(rotmat)
    qw = ti.sqrt(1.0 + tr) / 2.0
    qx = (m21 - m12) / (4.0 * qw)
    qy = (m02 - m20) / (4.0 * qw)
    qz = (m10 - m01) / (4.0 * qw)

    return ti.Vector([qw, qx, qy, qz])

@ti.func
def rotmat2rotvec(rotmat):
    m00, m01, m02 = rotmat[0, 0], rotmat[0, 1], rotmat[0, 2]
    m10, m11, m12 = rotmat[1, 0], rotmat[1, 1], rotmat[1, 2]
    m20, m21, m22 = rotmat[2, 0], rotmat[2, 1], rotmat[2, 2]

    tr = m00 + m11 + m22
    qw = ti.sqrt(1.0 + tr) / 2.0
    qx = (m21 - m12) / (4.0 * qw)
    qy = (m02 - m20) / (4.0 * qw)
    qz = (m10 - m01) / (4.0 * qw)

    angle = ti.acos((tr - 1.0) / 2.0)
    axis = (ti.Vector([qx, qy, qz]) * (ti.sin(angle) / abs(ti.sin(angle)) + 1e-9)).normalized()

    return angle * axis

@ti.kernel
def scan_aa_grid(euler_limit: ti.ext_arr()):

    x_min, x_max = euler_limit[0, 0], euler_limit[0, 1]
    y_min, y_max = euler_limit[1, 0], euler_limit[1, 1]
    z_min, z_max = euler_limit[2, 0], euler_limit[2, 1]

    for i, j, k in axis_angle:
        rx = x_min + (x_max - x_min) * i / (nx - 1.0)
        ry = y_min + (y_max - y_min) * j / (ny - 1.0)
        rz = z_min + (z_max - z_min) * k / (nz - 1.0)

        rmat = euler2rotmat(rx, ry, rz)
        rvec = rotmat2rotvec(rmat)
        vrgb = colormap(rvec)
        axis_angle[i, j, k] = rvec
        aa_colormp[i, j, k] = ti.Vector([vrgb[0], vrgb[1], vrgb[2], 0.0])

@ti.kernel
def animate_aa(t: ti.f32):

    rx, ry, rz = t * 0.0011, t * 0.0012, t * 0.0015
    rmat = euler2rotmat(rx, ry, rz)

    for I in ti.grouped(axis_angle):
        p_cam = rmat @ axis_angle[I] + ti.Vector([0.0, 0.0, t_pushed])
        depth = p_cam[2]
        p_img = K @ (p_cam / depth)
        p_img = ti.cast(p_img, ti.i32)
        aa_colormp[I][3] = depth
        aa_transfm[I] = ti.Vector([p_img[0], p_img[1]])
        
    # print("aa_tfm[0]", axis_angle[0, 0, 0], "->", aa_transfm[0, 0, 0], " << ", aa_colormp[0, 0, 0])

@ti.kernel
def init_pixels():

    for I in ti.grouped(pixels):
        pixels[I] = ti.Vector([0.0, 0.0, 0.0])
        z_buff[I] = ti.Vector([0.0, 0.0, 0.0, t_farest])

@ti.kernel
def render():

    # update z_buffer
    for I in ti.grouped(aa_colormp):
        depth = aa_colormp[I][3]
        px, py = aa_transfm[I][0], aa_transfm[I][1]
        if depth <= z_buff[px, py][3]:
            z_buff[px, py] = ti.Vector([aa_colormp[I][0], aa_colormp[I][1], aa_colormp[I][2], depth])

    # update screen
    for I in ti.grouped(pixels):
        pixels[I] = mix(pixels[I], ti.Vector([z_buff[I][0], z_buff[I][1], z_buff[I][2]]), 0.85)

def main():

    init_pixels()
    euler_range = np.array([
        [-0.6 * PI, +0.6 * PI],  # X >> [-Pi, +Pi]
        [-0.0 * PI, +0.0 * PI],  # Y >> [-.0, +.0]
        [-0.6 * PI, +0.6 * PI],  # Z >> [-Pi, +Pi]
    ], dtype=np.float64)
    scan_aa_grid(euler_range)

    gui = ti.GUI("LieAlgebra", res=(RES, RES))
    for ts in range(1000000):
        animate_aa(ts)
        render()
        gui.set_image(pixels.to_numpy())
        gui.show()

    # data = axis_angle.to_numpy().reshape((-1, 3))
    # # print(data.shape)
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(data[..., 0].reshape(-1), data[..., 1].reshape(-1), data[..., 2].reshape(-2), c=(data[..., 2] <= 0.0), cmap='coolwarm', marker='^')
    # plt.show()


if __name__ == "__main__":
    
    main()
