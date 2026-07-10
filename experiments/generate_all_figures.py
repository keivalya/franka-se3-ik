"""
ME5250 Project 2 — Generate all figures for the report.
Runs the full simulation pipeline and produces publication-quality plots.
"""
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, textwrap

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
CB_BLUE   = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN  = '#009E73'
CB_RED    = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_CYAN   = '#56B4E9'
CB_YELLOW = '#F0E442'
JOINT_COLORS = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, '#332288']

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'legend.fontsize': 8, 'legend.framealpha': 0.9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.8, 'grid.alpha': 0.3,
    'grid.color': '#CCCCCC', 'grid.linewidth': 0.5,
})

FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)

# =====================================================================
# 1. Load model, DH, helpers (same as run_simulation.py)
# =====================================================================
MODEL_PATH = 'mujoco_menagerie/franka_emika_panda/panda.xml'
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
NJ = 7; sid = 0
q_lo = model.jnt_range[:NJ,0].copy()
q_hi = model.jnt_range[:NJ,1].copy()
q_mid = 0.5*(q_lo+q_hi)

# Standard/Classical DH parameters (Lynch & Park 2017)
# Each row: [a_i, d_i, alpha_i, theta_offset_i]
DH = [[0,      0.333, -np.pi/2, 0],
      [0,      0,      np.pi/2, 0],
      [0.0825, 0.316,  np.pi/2, 0],
      [-0.0825,0,     -np.pi/2, 0],
      [0,      0.384,  np.pi/2, 0],
      [0.088,  0,      np.pi/2, 0],
      [0,      0.107,  0,       0]]

def dh_xf(a,d,al,th):
    ct,st,ca,sa=np.cos(th),np.sin(th),np.cos(al),np.sin(al)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0,   sa,     ca,    d   ],
                     [0,   0,      0,     1   ]])
def fk_chain(q):
    T=np.eye(4)
    for i in range(7): T=T@dh_xf(DH[i][0],DH[i][1],DH[i][2],q[i]+DH[i][3])
    return T

offsets=[]
for _ in range(100):
    qr=np.random.uniform(q_lo,q_hi)
    data.qpos[:NJ]=qr; mujoco.mj_forward(model,data)
    Tm=np.eye(4); Tm[:3,:3]=data.site_xmat[sid].reshape(3,3); Tm[:3,3]=data.site_xpos[sid]
    offsets.append(np.linalg.inv(fk_chain(qr))@Tm)
T_FE=np.mean(offsets,axis=0); T_FE[np.abs(T_FE)<1e-10]=0.0
def analytical_fk(q): return fk_chain(q)@T_FE

def se3_err(Td,Tc):
    ep=Td[:3,3]-Tc[:3,3]; Re=Td[:3,:3]@Tc[:3,:3].T
    eR=0.5*np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])
    return np.concatenate([ep,eR])
def ee_pose():
    T=np.eye(4); T[:3,:3]=data.site_xmat[sid].reshape(3,3); T[:3,3]=data.site_xpos[sid]; return T
def jac():
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,jp,jr,sid)
    return np.vstack([jp[:,:NJ],jr[:,:NJ]])
def solve_newton(T_des,q0,niter=5,nullspace=False,alpha=0.5):
    q=q0.copy()
    for _ in range(niter):
        data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
        e=se3_err(T_des,ee_pose()); J=jac(); Jp=np.linalg.pinv(J)
        dq=Jp@e
        if nullspace: dq+=(np.eye(NJ)-Jp@J)@(alpha*(q_mid-q))
        q=np.clip(q+dq,q_lo,q_hi)
    data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
    ef=se3_err(T_des,ee_pose()); Jf=jac()
    w=np.sqrt(max(0,np.linalg.det(Jf@Jf.T)))
    return q,ef,w,np.linalg.cond(Jf)
def solve_dls(T_des,q0,niter=5,lmax=0.05,w0=0.04):
    q=q0.copy()
    for _ in range(niter):
        data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
        e=se3_err(T_des,ee_pose()); J=jac()
        w=np.sqrt(max(0,np.linalg.det(J@J.T)))
        lam=lmax*(1-(w/w0)**2) if w<w0 else 0.0
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6),e)
        q=np.clip(q+dq,q_lo,q_hi)
    data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
    ef=se3_err(T_des,ee_pose()); Jf=jac()
    wf=np.sqrt(max(0,np.linalg.det(Jf@Jf.T)))
    return q,ef,wf

# =====================================================================
# 2. Trajectory
# =====================================================================
cx,cy,z0,r,h,ds = 0.4, 0.0, 0.3, 0.15, 0.1, 0.001
ds_per_rad = np.sqrt(r**2+(h/(2*np.pi))**2)
L = 2*np.pi*ds_per_rad; NW = int(np.ceil(L/ds))
sv = np.linspace(0,2*np.pi,NW)
Td = np.zeros((NW,4,4))
for i,s in enumerate(sv):
    px,py,pz = cx+r*np.cos(s), cy+r*np.sin(s), z0+(h/(2*np.pi))*s
    xh=np.array([-np.cos(s),-np.sin(s),0.]); zh=np.array([0,0,-1.]); yh=np.cross(zh,xh)
    Td[i]=np.eye(4); Td[i,:3,0]=xh; Td[i,:3,1]=yh; Td[i,:3,2]=zh; Td[i,:3,3]=[px,py,pz]
arc = np.linspace(0,L,NW)
print(f"Trajectory: L={L:.4f} m, N={NW}")

# =====================================================================
# 3. FK verification
# =====================================================================
print("Running FK verification...")
np.random.seed(42)
fk_epos,fk_erot=[],[]
for _ in range(200):
    qr=np.random.uniform(q_lo,q_hi)
    data.qpos[:NJ]=qr; mujoco.mj_forward(model,data)
    Tm=np.eye(4); Tm[:3,:3]=data.site_xmat[sid].reshape(3,3); Tm[:3,3]=data.site_xpos[sid]
    Ta=analytical_fk(qr)
    fk_epos.append(np.linalg.norm(Ta[:3,3]-Tm[:3,3]))
    fk_erot.append(np.linalg.norm(Ta[:3,:3]-Tm[:3,:3],'fro'))
fk_epos,fk_erot=np.array(fk_epos),np.array(fk_erot)

# =====================================================================
# 4. Solve IK — all three methods
# =====================================================================
print("Solving IK -- Newton (no null-space)...")
q_home = np.array([0,-0.3,0,-2.0,0,1.6,0.8])
q_init,_,_,_ = solve_newton(Td[0],q_home,niter=500,nullspace=False)
q_nn=np.zeros((NW,7)); e_nn=np.zeros((NW,6)); w_nn=np.zeros(NW); c_nn=np.zeros(NW)
qp=q_init.copy()
for i in range(NW):
    if i%200==0: print(f"  wp {i}/{NW}")
    q,e,w,c=solve_newton(Td[i],qp,5,False)
    q_nn[i]=q; e_nn[i]=e; w_nn[i]=w; c_nn[i]=c; qp=q.copy()

print("Solving IK -- Newton + null-space (alpha=0.5)...")
q_ns=np.zeros((NW,7)); e_ns=np.zeros((NW,6)); w_ns=np.zeros(NW)
qp=q_init.copy()
for i in range(NW):
    if i%200==0: print(f"  wp {i}/{NW}")
    q,e,w,_=solve_newton(Td[i],qp,5,True,0.5)
    q_ns[i]=q; e_ns[i]=e; w_ns[i]=w; qp=q.copy()

print("Solving IK -- DLS...")
q_dl=np.zeros((NW,7)); e_dl=np.zeros((NW,6)); w_dl=np.zeros(NW)
qp=q_init.copy()
for i in range(NW):
    if i%200==0: print(f"  wp {i}/{NW}")
    q,e,w=solve_dls(Td[i],qp,5)
    q_dl[i]=q; e_dl[i]=e; w_dl[i]=w; qp=q.copy()

pe_nn=np.linalg.norm(e_nn[:,:3],axis=1); oe_nn=np.linalg.norm(e_nn[:,3:],axis=1)
pe_dl=np.linalg.norm(e_dl[:,:3],axis=1); oe_dl=np.linalg.norm(e_dl[:,3:],axis=1)
pe_ns=np.linalg.norm(e_ns[:,:3],axis=1); oe_ns=np.linalg.norm(e_ns[:,3:],axis=1)
print(f"  Newton SS: pos={pe_nn[50:].mean()*1000:.6f} mm")

# =====================================================================
# Collect actual EE positions from solved q_nn (for path tracing)
# =====================================================================
print("Collecting EE positions for path trace...")
ee_positions = np.zeros((NW, 3))
for i in range(NW):
    data.qpos[:NJ] = q_nn[i]
    mujoco.mj_forward(model, data)
    ee_positions[i] = data.site_xpos[sid].copy()

# =====================================================================
# Scene XML builder — colorful, professional, attractive
# =====================================================================
def make_scene_xml():
    panda_abs = os.path.abspath(MODEL_PATH)
    panda_dir = os.path.dirname(panda_abs)
    with open(panda_abs, 'r') as f:
        xml_src = f.read()

    import re

    visual_block = textwrap.dedent("""\
      <visual>
        <global offwidth="1280" offheight="960"/>
        <quality shadowsize="4096"/>
        <map znear="0.01" zfar="10"/>
        <headlight diffuse="0.25 0.25 0.28" ambient="0.15 0.15 0.18" specular="0.08 0.08 0.08"/>
      </visual>
    """)

    world_inject = textwrap.dedent("""\
        <!-- Checker floor -->
        <geom name="floor" type="plane" size="2 2 0.01"
              pos="0 0 0" material="floor_mat"/>
        <!-- Key light — moderate warm -->
        <light name="key" pos="1.5 -1.2 2.2" dir="-0.5 0.4 -0.8"
               diffuse="0.45 0.40 0.32" specular="0.2 0.18 0.14" castshadow="true"/>
        <!-- Fill light — cool, gentle -->
        <light name="fill" pos="-1.2 1.0 1.8" dir="0.5 -0.3 -0.6"
               diffuse="0.22 0.25 0.35" specular="0.03 0.03 0.05" castshadow="false"/>
        <!-- Rim light — subtle -->
        <light name="rim" pos="-0.5 -0.8 2.8" dir="0.15 0.25 -1"
               diffuse="0.18 0.18 0.22" specular="0.06 0.06 0.08" castshadow="false"/>
        <!-- Fixed camera -->
        <camera name="nice_cam" pos="1.2 -1.0 0.85" xyaxes="0.65 0.76 0 -0.28 0.24 0.93"/>
    """)

    asset_inject = textwrap.dedent("""\
        <texture name="floor_tex" type="2d" builtin="checker"
                 rgb1="0.22 0.35 0.55" rgb2="0.85 0.55 0.25"
                 width="512" height="512"/>
        <material name="floor_mat" texture="floor_tex"
                  texrepeat="10 10" specular="0.3" shininess="0.5"
                  reflectance="0.15"/>
    """)

    xml_out = re.sub(r'(<mujoco[^>]*>)', r'\1\n'+visual_block, xml_src, count=1)
    xml_out = xml_out.replace('<worldbody>', '<worldbody>\n'+world_inject, 1)
    xml_out = xml_out.replace('<asset>', '<asset>\n'+asset_inject, 1)

    tmp_path = os.path.join(panda_dir, '_scene_tmp.xml')
    with open(tmp_path, 'w') as f:
        f.write(xml_out)
    return tmp_path

# =====================================================================
# FIGURE 1 — FINALIZED, do not regenerate
# =====================================================================
print("Skipping Fig 1 (finalized by user — do not overwrite)")

# =====================================================================
# FIGURE 2 — FINALIZED, do not regenerate
# =====================================================================
print("Skipping Fig 2 (finalized by user — do not overwrite)")

# =====================================================================
# FIGURE 3 — FK verification
# =====================================================================
print("Generating Fig 3: FK Verification...")
fig, ax = plt.subplots(figsize=(6, 3))
ax.semilogy(range(200), fk_epos, 'o', ms=3, color=CB_BLUE, alpha=0.7, label='Position error (m)')
ax.semilogy(range(200), fk_erot, 's', ms=3, color=CB_ORANGE, alpha=0.7, label='Rotation error (Frobenius)')
ax.axhline(2.2e-16, color='#888888', ls='--', lw=1.0, label=r'Machine $\varepsilon$')
ax.set_xlabel('Sample index'); ax.set_ylabel('Error (log scale)')
ax.set_ylim(1e-17, 1e-13)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, which='both', alpha=0.3, color='#CCCCCC', lw=0.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig3_fk_verification.png')
plt.close()
print("  Saved fig3_fk_verification.png")

# =====================================================================
# FIGURE 4 — IK tracking error
# =====================================================================
print("Generating Fig 4: IK Tracking Error...")
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6.5,4.5), sharex=True)
for a in (ax1,ax2):
    a.axvspan(0,arc[50],alpha=0.08,color='#888888')
    a.grid(True,which='both',alpha=0.3,color='#CCCCCC',lw=0.5)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
ax1.semilogy(arc, pe_nn*1000, '-', color=CB_BLUE, lw=1.0, label='Newton')
ax1.semilogy(arc, pe_dl*1000, '--', color=CB_ORANGE, lw=1.0, label='DLS')
ax1.fill_between(arc[50:], pe_nn[50:]*1000, alpha=0.12, color=CB_BLUE)
ax1.set_ylabel('Position error (mm)'); ax1.legend(fontsize=8, loc='upper right')
ax1.set_title('(a) Position Tracking Error', fontsize=11); ax1.set_ylim(1e-7,5e-1)
ax1.text(arc[25], ax1.get_ylim()[1]*0.3, 'convergence\nregion',
         ha='center', fontsize=7, color='#666666', style='italic')
ax2.semilogy(arc, np.degrees(oe_nn), '-', color=CB_BLUE, lw=1.0, label='Newton')
ax2.semilogy(arc, np.degrees(oe_dl), '--', color=CB_ORANGE, lw=1.0, label='DLS')
ax2.fill_between(arc[50:], np.degrees(oe_nn[50:]), alpha=0.12, color=CB_BLUE)
ax2.set_ylabel('Orientation error (deg)'); ax2.set_xlabel('Arc-length (m)')
ax2.set_title('(b) Orientation Tracking Error', fontsize=11)
ax2.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig4_ik_error.png')
plt.close()
print("  Saved fig4_ik_error.png")

# =====================================================================
# FIGURE 5 — Manipulability
# =====================================================================
print("Generating Fig 5: Manipulability...")
fig, ax1 = plt.subplots(figsize=(6.5, 3))
ax1.plot(arc, w_nn, '-', color=CB_GREEN, lw=1.4, label=r'$w(\mathbf{q})$')
ax1.axhline(0.04, color='#888888', ls='--', lw=1.0, label=r'$w_0 = 0.04$')
low_mask_p = w_nn < 0.04
ax1.fill_between(arc, 0, w_nn, where=low_mask_p, color=CB_RED, alpha=0.18, label='Low-manipulability zone')
mi = np.argmin(w_nn)
ax1.annotate(f'$w_{{\\min}}$ = {w_nn[mi]:.4f}', xy=(arc[mi], w_nn[mi]),
             xytext=(0.42, 0.083),
             arrowprops=dict(arrowstyle='->', color=CB_RED, lw=1.5),
             fontsize=9, color=CB_RED, fontweight='bold')
ax1.set_xlabel('Arc-length (m)'); ax1.set_ylabel(r'Manipulability $w(\mathbf{q})$')
ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax1.grid(True,alpha=0.3,color='#CCCCCC',lw=0.5); ax1.spines['top'].set_visible(False)
ax2 = ax1.twinx()
ax2.plot(arc, c_nn, '-', color=CB_PURPLE, lw=0.7, alpha=0.5)
ax2.set_ylabel(r'Condition number $\kappa(J)$', color=CB_PURPLE)
ax2.tick_params(axis='y', labelcolor=CB_PURPLE); ax2.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig5_manipulability.png')
plt.close()
print("  Saved fig5_manipulability.png")

# =====================================================================
# FIGURE 6 — Null-space joint trajectories
# =====================================================================
print("Generating Fig 6: Null-Space Effect...")
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8,3.8), sharey=True)
for j in range(7):
    c = JOINT_COLORS[j]; lbl = f'$q_{j+1}$'
    ax1.plot(arc, np.degrees(q_nn[:,j]), color=c, lw=0.9, label=lbl)
    ax1.axhline(np.degrees(q_lo[j]), color=c, ls='--', lw=0.7, alpha=0.45)
    ax1.axhline(np.degrees(q_hi[j]), color=c, ls='--', lw=0.7, alpha=0.45)
    ax2.plot(arc, np.degrees(q_ns[:,j]), color=c, lw=0.9)
    ax2.axhline(np.degrees(q_lo[j]), color=c, ls='--', lw=0.7, alpha=0.45)
    ax2.axhline(np.degrees(q_hi[j]), color=c, ls='--', lw=0.7, alpha=0.45)
ax1.set_title('(a) Without Null-Space Projection',fontsize=11)
ax1.set_xlabel('Arc-length (m)'); ax1.set_ylabel('Joint angle (deg)')
ax1.legend(fontsize=7,ncol=2,loc='lower left',framealpha=0.85)
ax1.grid(True,alpha=0.3,color='#CCCCCC',lw=0.5)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax2.set_title(r'(b) With Null-Space ($\alpha = 0.5$)',fontsize=11)
ax2.set_xlabel('Arc-length (m)')
ax2.grid(True,alpha=0.3,color='#CCCCCC',lw=0.5)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig6_nullspace.png')
plt.close()
print("  Saved fig6_nullspace.png")

# =====================================================================
# FIGURE 7 — MuJoCo filmstrip with EE path tracing (2x5, 10 frames)
# =====================================================================
print("Generating Fig 7: MuJoCo filmstrip with path trace...")
try:
    scene_xml = make_scene_xml()
    m7 = mujoco.MjModel.from_xml_path(scene_xml)
    d7 = mujoco.MjData(m7)

    W, H = 640, 480
    renderer = mujoco.Renderer(m7, height=H, width=W)
    cam_id7 = mujoco.mj_name2id(m7, mujoco.mjtObj.mjOBJ_CAMERA, 'nice_cam')

    pct = [0, 11, 22, 33, 44, 55, 66, 77, 88, 100]
    frame_indices = [min(int(p/100.0*(NW-1)+0.5), NW-1) for p in pct]

    # Build camera matrices for 3D->2D projection of the path
    # We'll use MuJoCo's scene to project world points to pixels
    def world_to_pixel(renderer, model, data, cam_id, pts_3d):
        """Project 3D world points to 2D pixel coords using MuJoCo scene camera."""
        scene = renderer._scene
        cam = scene.camera[0]  # main camera

        # Camera transform
        cam_pos = np.array(cam.pos)
        cam_fwd = np.array(cam.forward)
        cam_up  = np.array(cam.up)
        cam_right = np.cross(cam_fwd, cam_up)

        # View matrix (world -> camera)
        R = np.array([cam_right, -cam_up, cam_fwd])  # rows
        t = -R @ cam_pos

        # Perspective: fovy from model
        fovy = model.vis.global_.fovy if model.vis.global_.fovy > 0 else 45.0
        aspect = W / H
        fovy_rad = np.radians(fovy)
        fy = 1.0 / np.tan(fovy_rad / 2.0)
        fx = fy / aspect

        pixels = []
        for p3 in pts_3d:
            pc = R @ p3 + t  # camera coords
            if pc[2] < 0.01:  # behind camera
                pixels.append(None)
                continue
            px = fx * pc[0] / pc[2]
            py = fy * pc[1] / pc[2]
            # NDC -> pixel
            u = (px + 1.0) * 0.5 * W
            v = (py + 1.0) * 0.5 * H
            pixels.append((u, v))
        return pixels

    frames = []
    path_overlays = []  # list of pixel-coord arrays for path up to each frame

    for k, idx in enumerate(frame_indices):
        d7.qpos[:NJ] = q_nn[idx]
        mujoco.mj_forward(m7, d7)
        renderer.update_scene(d7, camera=cam_id7)
        raw_img = renderer.render().copy()

        # Project EE path up to this waypoint onto the image
        trail_pts = ee_positions[:idx+1:max(1, idx//150)]  # subsample for speed
        pixels = world_to_pixel(renderer, m7, d7, cam_id7, trail_pts)

        # Draw the path on the image as an overlay
        img_overlay = raw_img.copy()
        valid = [(int(round(u)), int(round(v))) for p in pixels
                 if p is not None for u, v in [p]
                 if 0 <= int(round(u)) < W and 0 <= int(round(v)) < H]

        # Draw thick orange trail
        for i_px in range(len(valid)-1):
            u0,v0 = valid[i_px]
            u1,v1 = valid[i_px+1]
            # Bresenham-ish thick line
            nsteps = max(abs(u1-u0), abs(v1-v0), 1)
            for t_step in range(nsteps+1):
                frac = t_step / nsteps
                uu = int(round(u0 + frac*(u1-u0)))
                vv = int(round(v0 + frac*(v1-v0)))
                for du in range(-2, 3):
                    for dv in range(-2, 3):
                        ui, vi = uu+du, vv+dv
                        if 0 <= ui < W and 0 <= vi < H:
                            img_overlay[vi, ui] = [220, 20, 60]  # crimson red

        frames.append(img_overlay)

    renderer.close()
    os.unlink(scene_xml)

    # Compose filmstrip
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
    for k, (ax, frame, p) in enumerate(zip(axes.flat, frames, pct)):
        ax.imshow(frame)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(f'{FIGDIR}/fig7_mujoco_composite.png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print("  Saved fig7_mujoco_composite.png")
except Exception as ex:
    print(f"  WARNING fig7: {ex}")
    import traceback; traceback.print_exc()

# =====================================================================
# FIGURE 8 — Reachable workspace (no in-image title)
# =====================================================================
print("Generating Fig 8: Workspace...")
np.random.seed(7)
n_ws = 8000
ws_pts = np.zeros((n_ws, 3))
for k in range(n_ws):
    qr = np.random.uniform(q_lo, q_hi)
    data.qpos[:NJ] = qr; mujoco.mj_forward(model, data)
    ws_pts[k] = data.site_xpos[sid].copy()

traj_pts = ee_positions  # actual solved EE path

fig, (axXZ, axXY) = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, (xi, yi, xl, yl) in zip(
        [axXZ, axXY],
        [(0, 2, 'X (m)', 'Z (m)'), (0, 1, 'X (m)', 'Y (m)')]):
    ax.scatter(ws_pts[:, xi], ws_pts[:, yi],
               s=1.5, c=CB_CYAN, alpha=0.25, linewidths=0, rasterized=True)
    ax.plot(traj_pts[:, xi], traj_pts[:, yi],
            color=CB_ORANGE, lw=1.4, zorder=3)
    ax.scatter(traj_pts[0,  xi], traj_pts[0,  yi],
               color=CB_GREEN, s=35, zorder=5, edgecolors='k', lw=0.6)
    ax.scatter(traj_pts[-1, xi], traj_pts[-1, yi],
               color=CB_RED,   s=35, zorder=5, edgecolors='k', lw=0.6)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3, color='#CCCCCC', lw=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

axXZ.set_title('(a) XZ projection', fontsize=10)
axXY.set_title('(b) XY projection', fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig8_workspace.png')
plt.close()
print("  Saved fig8_workspace.png")

# =====================================================================
# FIGURE 9 — 3D velocity ellipsoids overlaid on MuJoCo renders
#   Per-panel free camera zoomed tight to each EE. White background via
#   BFS flood-fill from image corners. Lynch & Park style.
# =====================================================================
print("Generating Fig 9: Velocity ellipsoids (MuJoCo renders)...")

def _make_scene_xml_fig9():
    """Minimal scene: no floor/walls, balanced lighting, no shadows."""
    import re as _re2
    panda_abs = os.path.abspath(MODEL_PATH)
    panda_dir = os.path.dirname(panda_abs)
    with open(panda_abs, 'r') as f:
        xml_src = f.read()
    visual_block = textwrap.dedent("""\
      <visual>
        <global offwidth="960" offheight="960"/>
        <quality shadowsize="0"/>
        <map znear="0.01" zfar="5"/>
        <headlight diffuse="0.55 0.55 0.55" ambient="0.60 0.60 0.60"
                   specular="0.05 0.05 0.05"/>
      </visual>\n""")
    world_inject = textwrap.dedent("""\
        <light name="ell_key"  pos="0.8 -0.6 1.2" dir="-0.4 0.3 -0.8"
               diffuse="0.55 0.52 0.45" castshadow="false"/>
        <light name="ell_fill" pos="-0.6 0.5 0.9" dir="0.4 -0.2 -0.6"
               diffuse="0.30 0.32 0.42" castshadow="false"/>
        <light name="ell_top"  pos="0.4 0.0 1.5" dir="0.0 0.0 -1.0"
               diffuse="0.25 0.25 0.28" castshadow="false"/>\n""")
    xml_out = _re2.sub(r'(<mujoco[^>]*>)', r'\1\n' + visual_block, xml_src, count=1)
    xml_out = xml_out.replace('<worldbody>', '<worldbody>\n' + world_inject, 1)
    tmp = os.path.join(panda_dir, '_scene_fig9.xml')
    with open(tmp, 'w') as f: f.write(xml_out)
    return tmp

def _bg_to_white(img, thresh=22):
    """BFS flood-fill from corners: replace true sky (max channel < thresh) with white.
    Uses max-channel threshold to avoid eating dark robot joint geometry."""
    from collections import deque
    H, W = img.shape[:2]
    candidate = img.max(axis=2) < thresh   # sky ≈ 0–8; dark robot joints ≈ 30+
    visited = np.zeros((H, W), bool)
    q = deque()
    for r in range(H):
        for c in (0, W-1):
            if candidate[r, c] and not visited[r, c]:
                visited[r, c] = True; q.append((r, c))
    for c in range(W):
        for r in (0, H-1):
            if candidate[r, c] and not visited[r, c]:
                visited[r, c] = True; q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr,nc] and candidate[nr,nc]:
                visited[nr,nc] = True; q.append((nr, nc))
    out = img.copy()
    out[visited] = 255
    return out

scene_xml9 = _make_scene_xml_fig9()
m9 = mujoco.MjModel.from_xml_path(scene_xml9)
d9 = mujoco.MjData(m9)
sid9 = mujoco.mj_name2id(m9, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
RW9, RH9 = 520, 520
renderer9 = mujoco.Renderer(m9, height=RH9, width=RW9)

ELL_WPS  = [0, 234, 468, 702, 947]
ELL_RGBA = np.array([0.05, 0.72, 0.95, 0.42], np.float32)   # bright cyan, semi-transparent
AXIS_RGBA = [
    np.array([1.00, 0.10, 0.10, 1.0], np.float32),  # red   — σ₁
    np.array([0.08, 0.90, 0.08, 1.0], np.float32),  # green — σ₂
    np.array([0.15, 0.40, 1.00, 1.0], np.float32),  # blue  — σ₃
]
EE_RGBA = np.array([1.0, 0.80, 0.0, 1.0], np.float32)

def _rot_z_to_v(v):
    v = np.asarray(v, float); v /= np.linalg.norm(v)
    z = np.array([0., 0., 1.]); dot = float(np.dot(z, v))
    if dot >  0.9999: return np.eye(3)
    if dot < -0.9999: return np.diag([1., -1., -1.])
    axis = np.cross(z, v); axis /= np.linalg.norm(axis)
    ang = np.arccos(np.clip(dot, -1., 1.))
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)

def _add_geom(scene, gtype, size, pos, mat, rgba):
    n = scene.ngeom
    if n >= scene.maxgeom: return
    mujoco.mjv_initGeom(scene.geoms[n], gtype,
                        np.asarray(size, np.float64), np.asarray(pos, np.float64),
                        np.asarray(mat,  np.float64).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1

# global scale: max semi-axis = 5.5 cm — ellipsoid prominent but arm still visible
max_s1 = 0.0
for wp in ELL_WPS:
    d9.qpos[:NJ] = q_nn[wp]; mujoco.mj_forward(m9, d9)
    jp9 = np.zeros((3, m9.nv)); jr9 = np.zeros((3, m9.nv))
    mujoco.mj_jacSite(m9, d9, jp9, jr9, sid9)
    max_s1 = max(max_s1, np.linalg.svd(jp9[:, :NJ], compute_uv=False)[0])
ELL_SCALE = 0.16 / max_s1   # exaggerated for visual clarity

panels9, w_vals9 = [], []
for wp in ELL_WPS:
    d9.qpos[:NJ] = q_nn[wp]; mujoco.mj_forward(m9, d9)
    ee_p = d9.site_xpos[sid9].copy()

    jp9 = np.zeros((3, m9.nv)); jr9 = np.zeros((3, m9.nv))
    mujoco.mj_jacSite(m9, d9, jp9, jr9, sid9)
    Jp9 = jp9[:, :NJ]
    U9, S9, _ = np.linalg.svd(Jp9, full_matrices=False)
    if np.linalg.det(U9) < 0: U9[:, 2] *= -1
    w_vals9.append(float(np.sqrt(max(0., np.linalg.det(Jp9 @ Jp9.T)))))

    # per-panel free camera: lock onto this EE, close enough to fill frame with wrist
    cam9 = mujoco.MjvCamera()
    cam9.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam9.lookat[:] = ee_p
    cam9.distance  = 0.72   # show full arm from base to EE
    cam9.azimuth   = 215.0
    cam9.elevation = 18.0

    renderer9.update_scene(d9, camera=cam9)
    sc9 = renderer9.scene

    # ellipsoid (semi-transparent)
    _add_geom(sc9, mujoco.mjtGeom.mjGEOM_ELLIPSOID, S9 * ELL_SCALE, ee_p, U9, ELL_RGBA)

    # axis spikes: two cylinders per axis, each outside the ellipsoid surface
    AX_R = 0.009
    for i, ax_rgba in enumerate(AXIS_RGBA):
        half_len  = float(S9[i] * ELL_SCALE)
        direction = U9[:, i]
        for sign in (1., -1.):
            cyl_center = ee_p + sign * direction * 1.65 * half_len
            _add_geom(sc9, mujoco.mjtGeom.mjGEOM_CYLINDER,
                      [AX_R, 0.65 * half_len, 0.], cyl_center,
                      _rot_z_to_v(sign * direction), ax_rgba)

    _add_geom(sc9, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0., 0.], ee_p, np.eye(3), EE_RGBA)

    panels9.append(renderer9.render().copy())

renderer9.close()
os.unlink(scene_xml9)

# compose — white figure (report style), MuJoCo panels keep their dark sim background
from matplotlib.patches import Patch as _Patch
fig9, axes9 = plt.subplots(1, 5, figsize=(15.5, 3.8))
fig9.patch.set_facecolor('white')
for ax9, wp, img9, w9 in zip(axes9, ELL_WPS, panels9, w_vals9):
    ax9.imshow(img9)
    ax9.axis('off')
    ax9.set_title(f'$s$={arc[wp]:.2f} m,  $w$={w9:.4f}', fontsize=8.5, pad=3)

legend_els9 = [
    _Patch(facecolor='#12B8F2', alpha=0.7, label='Velocity ellipsoid'),
    _Patch(facecolor='#FF1A1A', label='$\\sigma_1$ (major)'),
    _Patch(facecolor='#14E614', label='$\\sigma_2$'),
    _Patch(facecolor='#2666FF', label='$\\sigma_3$ (minor)'),
    _Patch(facecolor='#FFCC00', label='EE centre'),
]
fig9.legend(handles=legend_els9, loc='lower center', ncol=5,
            fontsize=8.5, framealpha=0.9, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, 0.0), bbox_transform=fig9.transFigure)
fig9.subplots_adjust(bottom=0.11, wspace=0.03, top=0.93, left=0.01, right=0.99)
plt.savefig(f'{FIGDIR}/fig9_ellipsoids.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved fig9_ellipsoids.png")

# =====================================================================
print("\n" + "="*55)
print("ALL FIGURES GENERATED")
print("="*55)
for f in sorted(os.listdir(FIGDIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(f'{FIGDIR}/{f}')/1024
        print(f"  {f:42s} {size_kb:6.0f} KB")
print("\nDone!")
