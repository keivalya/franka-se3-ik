"""
ME5250 Project 2 — Franka Panda 7-DOF SE(3) IK Solver
Final data collection script. Uses home config warm-start.
"""
import numpy as np
import mujoco
import json

MODEL_PATH = 'mujoco_menagerie/franka_emika_panda/panda.xml'
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
NJ = 7; sid = 0  # attachment_site
q_lo = model.jnt_range[:NJ, 0].copy()
q_hi = model.jnt_range[:NJ, 1].copy()
q_mid = 0.5 * (q_lo + q_hi)

# ---- DH parameters (Standard/Classical convention, Lynch & Park 2017) ----
# Each row: [a_i, d_i, alpha_i, theta_offset_i]
# T_i = Rz(theta)*Tz(d)*Tx(a)*Rx(alpha)
DH = [[0,      0.333, -np.pi/2, 0],
      [0,      0,      np.pi/2, 0],
      [0.0825, 0.316,  np.pi/2, 0],
      [-0.0825,0,     -np.pi/2, 0],
      [0,      0.384,  np.pi/2, 0],
      [0.088,  0,      np.pi/2, 0],
      [0,      0.107,  0,       0]]

def dh_xf(a,d,al,th):
    ct,st,ca,sa = np.cos(th),np.sin(th),np.cos(al),np.sin(al)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0,   sa,     ca,    d   ],
                     [0,   0,      0,     1   ]])

def fk_chain(q):
    T = np.eye(4)
    for i in range(7): T = T @ dh_xf(DH[i][0],DH[i][1],DH[i][2],q[i]+DH[i][3])
    return T

# ---- Flange-to-EE offset ----
offsets = []
for _ in range(100):
    qr = np.random.uniform(q_lo, q_hi)
    data.qpos[:NJ]=qr; mujoco.mj_forward(model,data)
    Tm=np.eye(4); Tm[:3,:3]=data.site_xmat[sid].reshape(3,3); Tm[:3,3]=data.site_xpos[sid]
    offsets.append(np.linalg.inv(fk_chain(qr))@Tm)
T_FE = np.mean(offsets,axis=0); T_FE[np.abs(T_FE)<1e-10]=0.0
angle_off = np.degrees(np.arccos(np.clip((np.trace(T_FE[:3,:3])-1)/2,-1,1)))
print(f"Flange-to-EE: {angle_off:.1f}deg rotation about z-axis")
def analytical_fk(q): return fk_chain(q) @ T_FE

# ---- FK Verification ----
epos_list, erot_list = [], []
for _ in range(200):
    qr=np.random.uniform(q_lo,q_hi)
    data.qpos[:NJ]=qr; mujoco.mj_forward(model,data)
    Tm=np.eye(4); Tm[:3,:3]=data.site_xmat[sid].reshape(3,3); Tm[:3,3]=data.site_xpos[sid]
    Ta=analytical_fk(qr)
    epos_list.append(np.linalg.norm(Ta[:3,3]-Tm[:3,3]))
    erot_list.append(np.linalg.norm(Ta[:3,:3]-Tm[:3,:3],'fro'))
epos_a, erot_a = np.array(epos_list), np.array(erot_list)
print(f"FK: pos mean={epos_a.mean():.2e} max={epos_a.max():.2e}, rot mean={erot_a.mean():.2e} max={erot_a.max():.2e}")

# ---- Trajectory ----
cx,cy,z0,r,h,ds = 0.4, 0.0, 0.3, 0.15, 0.1, 0.001
ds_per_rad = np.sqrt(r**2+(h/(2*np.pi))**2)
L = 2*np.pi*ds_per_rad
NW = int(np.ceil(L/ds))
sv = np.linspace(0,2*np.pi,NW)
Td = np.zeros((NW,4,4))
for i,s in enumerate(sv):
    px,py,pz = cx+r*np.cos(s), cy+r*np.sin(s), z0+(h/(2*np.pi))*s
    xh=np.array([-np.cos(s),-np.sin(s),0.]); zh=np.array([0,0,-1.]); yh=np.cross(zh,xh)
    Td[i]=np.eye(4); Td[i,:3,0]=xh; Td[i,:3,1]=yh; Td[i,:3,2]=zh; Td[i,:3,3]=[px,py,pz]
print(f"Traj: L={L:.4f}m, N={NW}")

# ---- Helpers ----
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

def solve_newton(T_des, q0, niter=5, nullspace=False, alpha=0.5):
    q=q0.copy()
    for _ in range(niter):
        data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
        e=se3_err(T_des,ee_pose()); J=jac(); Jp=np.linalg.pinv(J)
        dq = Jp@e
        if nullspace: dq += (np.eye(NJ)-Jp@J)@(alpha*(q_mid-q))
        q=np.clip(q+dq,q_lo,q_hi)
    data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
    ef=se3_err(T_des,ee_pose()); Jf=jac()
    w=np.sqrt(max(0,np.linalg.det(Jf@Jf.T))); c=np.linalg.cond(Jf)
    return q, ef, w, c

def solve_dls(T_des, q0, niter=5, lmax=0.05, w0=0.04):
    q=q0.copy()
    for _ in range(niter):
        data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
        e=se3_err(T_des,ee_pose()); J=jac()
        w=np.sqrt(max(0,np.linalg.det(J@J.T)))
        lam = lmax*(1-(w/w0)**2) if w<w0 else 0.0
        dq = J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6),e)
        q=np.clip(q+dq,q_lo,q_hi)
    data.qpos[:NJ]=q; mujoco.mj_forward(model,data)
    ef=se3_err(T_des,ee_pose()); Jf=jac()
    wf=np.sqrt(max(0,np.linalg.det(Jf@Jf.T)))
    return q, ef, wf

# ---- Initialize from home config ----
q_home = np.array([0, -0.3, 0, -2.0, 0, 1.6, 0.8])
q_init, e_init = solve_newton(Td[0], q_home, niter=500, nullspace=False)[:2]
print(f"WP0 init: pos={np.linalg.norm(e_init[:3])*1000:.4f}mm ori={np.degrees(np.linalg.norm(e_init[3:])):.4f}deg")

# ---- Run all three methods ----
print("Running Newton (no null-space)...")
q_nn=np.zeros((NW,7)); e_nn=np.zeros((NW,6)); w_nn=np.zeros(NW); c_nn=np.zeros(NW)
qp=q_init.copy()
for i in range(NW):
    q,e,w,c = solve_newton(Td[i],qp,5,False)
    q_nn[i]=q; e_nn[i]=e; w_nn[i]=w; c_nn[i]=c; qp=q.copy()

print("Running Newton + null-space (alpha=0.5)...")
q_ns=np.zeros((NW,7)); e_ns=np.zeros((NW,6)); w_ns=np.zeros(NW); c_ns=np.zeros(NW)
qp=q_init.copy()
for i in range(NW):
    q,e,w,c = solve_newton(Td[i],qp,5,True,0.5)
    q_ns[i]=q; e_ns[i]=e; w_ns[i]=w; c_ns[i]=c; qp=q.copy()

# Check if null-space alpha=0.5 diverges
oe_ns_test = np.degrees(np.linalg.norm(e_ns[:,3:],axis=1))
if oe_ns_test.mean() > 5.0:
    print("  alpha=0.5 diverges, trying alpha=0.1...")
    qp=q_init.copy()
    for i in range(NW):
        q,e,w,c = solve_newton(Td[i],qp,5,True,0.1)
        q_ns[i]=q; e_ns[i]=e; w_ns[i]=w; c_ns[i]=c; qp=q.copy()
    ns_alpha = 0.1
else:
    ns_alpha = 0.5

print(f"Running DLS (w0=0.04)...")
q_dl=np.zeros((NW,7)); e_dl=np.zeros((NW,6)); w_dl=np.zeros(NW)
qp=q_init.copy()
for i in range(NW):
    q,e,w = solve_dls(Td[i],qp,5)
    q_dl[i]=q; e_dl[i]=e; w_dl[i]=w; qp=q.copy()

# ---- Compute stats ----
arc = np.linspace(0,L,NW)
pe_nn = np.linalg.norm(e_nn[:,:3],axis=1)
oe_nn = np.linalg.norm(e_nn[:,3:],axis=1)
pe_ns = np.linalg.norm(e_ns[:,:3],axis=1)
oe_ns = np.linalg.norm(e_ns[:,3:],axis=1)
pe_dl = np.linalg.norm(e_dl[:,:3],axis=1)
oe_dl = np.linalg.norm(e_dl[:,3:],axis=1)

# Steady-state (after convergence from home config, wp >= 50)
SS = 50

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nFK Verification (200 random configs):")
print(f"  Pos error: mean={epos_a.mean():.2e} m, max={epos_a.max():.2e} m")
print(f"  Rot error: mean={erot_a.mean():.2e}, max={erot_a.max():.2e}")

print(f"\nTrajectory: L={L:.4f}m, N={NW}, helix(cx={cx},cy={cy},z0={z0},r={r},h={h})")

for label, pe, oe in [("Newton",pe_nn,oe_nn),("Newton+NS",pe_ns,oe_ns),("DLS",pe_dl,oe_dl)]:
    print(f"\n{label} — Full trajectory:")
    print(f"  pos: mean={pe.mean()*1000:.4f}mm max={pe.max()*1000:.4f}mm")
    print(f"  ori: mean={np.degrees(oe.mean()):.4f}deg max={np.degrees(oe.max()):.4f}deg")
    print(f"  Steady-state (wp>={SS}):")
    print(f"    pos: mean={pe[SS:].mean()*1000:.6f}mm max={pe[SS:].max()*1000:.6f}mm")
    print(f"    ori: mean={np.degrees(oe[SS:].mean()):.6f}deg max={np.degrees(oe[SS:].max()):.6f}deg")

mi = np.argmin(w_nn)
print(f"\nManipulability (Newton):")
print(f"  min={w_nn.min():.6f} at wp{mi} (arc={arc[mi]:.4f}m)")
print(f"  max={w_nn.max():.6f}, mean={w_nn.mean():.6f}")
print(f"  w < 0.04: {(w_nn<0.04).sum()} waypoints ({(w_nn<0.04).sum()/NW*100:.1f}%)")
print(f"  Cond: mean={c_nn.mean():.1f} max={c_nn.max():.1f} min={c_nn.min():.1f}")

# Null-space effect
dnn = np.abs(q_nn - q_mid); dns = np.abs(q_ns - q_mid)
mnn,mns = dnn.mean(), dns.mean()
red = (1-mns/mnn)*100 if mnn>0 else 0
print(f"\nNull-Space Effect (alpha={ns_alpha}):")
print(f"  Mean dev from midpoint: no-null={np.degrees(mnn):.1f}deg, with-null={np.degrees(mns):.1f}deg")
print(f"  Reduction: {red:.1f}%")
for j in range(7):
    dn_j = dnn[:,j].max(); ds_j = dns[:,j].max()
    chg = (1-ds_j/dn_j)*100 if dn_j>0 else 0
    print(f"  J{j+1}: no-null max dev={np.degrees(dn_j):.1f}deg, with-null={np.degrees(ds_j):.1f}deg ({chg:+.1f}%)")

# DLS vs Newton comparison in low-manip regions
low_mask = w_nn < 0.04
if low_mask.sum() > 0:
    print(f"\nDLS vs Newton in low-manip regions (w<0.04, {low_mask.sum()} waypoints):")
    print(f"  Newton pos: mean={pe_nn[low_mask].mean()*1000:.4f}mm max={pe_nn[low_mask].max()*1000:.4f}mm")
    print(f"  DLS pos:    mean={pe_dl[low_mask].mean()*1000:.4f}mm max={pe_dl[low_mask].max()*1000:.4f}mm")
    print(f"  Newton ori: mean={np.degrees(oe_nn[low_mask].mean()):.4f}deg")
    print(f"  DLS ori:    mean={np.degrees(oe_dl[low_mask].mean()):.4f}deg")

# Save JSON
results = {
    'fk_pos_mean': float(epos_a.mean()), 'fk_pos_max': float(epos_a.max()),
    'fk_rot_mean': float(erot_a.mean()), 'fk_rot_max': float(erot_a.max()),
    'L': float(L), 'NW': NW,
    'helix_cx':cx,'helix_cy':cy,'helix_z0':z0,'helix_r':r,'helix_h':h,
    'nn_pos_mean_mm': float(pe_nn.mean()*1000), 'nn_pos_max_mm': float(pe_nn.max()*1000),
    'nn_ori_mean_deg': float(np.degrees(oe_nn.mean())), 'nn_ori_max_deg': float(np.degrees(oe_nn.max())),
    'nn_ss_pos_mean_mm': float(pe_nn[SS:].mean()*1000), 'nn_ss_pos_max_mm': float(pe_nn[SS:].max()*1000),
    'nn_ss_ori_mean_deg': float(np.degrees(oe_nn[SS:].mean())), 'nn_ss_ori_max_deg': float(np.degrees(oe_nn[SS:].max())),
    'ns_pos_mean_mm': float(pe_ns.mean()*1000), 'ns_pos_max_mm': float(pe_ns.max()*1000),
    'ns_ori_mean_deg': float(np.degrees(oe_ns.mean())), 'ns_ori_max_deg': float(np.degrees(oe_ns.max())),
    'ns_ss_pos_mean_mm': float(pe_ns[SS:].mean()*1000), 'ns_ss_pos_max_mm': float(pe_ns[SS:].max()*1000),
    'ns_ss_ori_mean_deg': float(np.degrees(oe_ns[SS:].mean())), 'ns_ss_ori_max_deg': float(np.degrees(oe_ns[SS:].max())),
    'ns_alpha': ns_alpha,
    'dl_pos_mean_mm': float(pe_dl.mean()*1000), 'dl_pos_max_mm': float(pe_dl.max()*1000),
    'dl_ori_mean_deg': float(np.degrees(oe_dl.mean())), 'dl_ori_max_deg': float(np.degrees(oe_dl.max())),
    'dl_ss_pos_mean_mm': float(pe_dl[SS:].mean()*1000), 'dl_ss_pos_max_mm': float(pe_dl[SS:].max()*1000),
    'dl_ss_ori_mean_deg': float(np.degrees(oe_dl[SS:].mean())), 'dl_ss_ori_max_deg': float(np.degrees(oe_dl[SS:].max())),
    'w_min': float(w_nn.min()), 'w_max': float(w_nn.max()), 'w_mean': float(w_nn.mean()),
    'w_min_idx': int(mi), 'w_min_arc': float(arc[mi]),
    'cond_mean': float(c_nn.mean()), 'cond_max': float(c_nn.max()), 'cond_min': float(c_nn.min()),
    'ns_red_pct': float(red),
    'n_low_manip_04': int(low_mask.sum()),
    'flange_deg': float(angle_off),
    'convergence_waypoints': SS,
}
with open('simulation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to simulation_results.json")
