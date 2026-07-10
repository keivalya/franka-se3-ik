"""
ME5250 Project 2 — Detailed Data Exporter
Runs the simulations and exports step-by-step data for the interactive charts.
"""
import numpy as np
import mujoco
import json
import os

MODEL_PATH = 'mujoco_menagerie/franka_emika_panda/panda.xml'
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
NJ = 7; sid = 0  # attachment_site
q_lo = model.jnt_range[:NJ, 0].copy()
q_hi = model.jnt_range[:NJ, 1].copy()
q_mid = 0.5 * (q_lo + q_hi)

# ---- DH parameters ----
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

# Flange-to-EE offset
offsets = []
for _ in range(100):
    qr = np.random.uniform(q_lo, q_hi)
    data.qpos[:NJ]=qr; mujoco.mj_forward(model,data)
    Tm=np.eye(4); Tm[:3,:3]=data.site_xmat[sid].reshape(3,3); Tm[:3,3]=data.site_xpos[sid]
    offsets.append(np.linalg.inv(fk_chain(qr))@Tm)
T_FE = np.mean(offsets,axis=0); T_FE[np.abs(T_FE)<1e-10]=0.0

def analytical_fk(q): return fk_chain(q) @ T_FE

# Trajectory
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
    q = q0.copy()
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
    q = q0.copy()
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

q_home = np.array([0, -0.3, 0, -2.0, 0, 1.6, 0.8])
q_init, _ = solve_newton(Td[0], q_home, niter=500, nullspace=False)[:2]

# Run Methods
arc = np.linspace(0, L, NW)
export_data = {
    "arc": arc.tolist(),
    "x_des": Td[:, 0, 3].tolist(),
    "y_des": Td[:, 1, 3].tolist(),
    "z_des": Td[:, 2, 3].tolist(),
    "q_min": q_lo.tolist(),
    "q_max": q_hi.tolist(),
    "q_mid": q_mid.tolist(),
    "methods": {}
}

# 1. Newton (no null-space)
print("Simulating Newton (no null-space)...")
q_nn, pe_nn, oe_nn, w_nn, c_nn = [], [], [], [], []
qp = q_init.copy()
for i in range(NW):
    q, e, w, c = solve_newton(Td[i], qp, 5, False)
    q_nn.append(q.tolist())
    pe_nn.append(float(np.linalg.norm(e[:3]) * 1000))  # mm
    oe_nn.append(float(np.degrees(np.linalg.norm(e[3:]))))  # deg
    w_nn.append(float(w))
    c_nn.append(float(c))
    qp = q.copy()

export_data["methods"]["newton"] = {
    "q": q_nn,
    "pe": pe_nn,
    "oe": oe_nn,
    "w": w_nn,
    "cond": c_nn
}

# 2. DLS
print("Simulating DLS...")
q_dl, pe_dl, oe_dl, w_dl = [], [], [], []
qp = q_init.copy()
for i in range(NW):
    q, e, w = solve_dls(Td[i], qp, 5)
    q_dl.append(q.tolist())
    pe_dl.append(float(np.linalg.norm(e[:3]) * 1000))
    oe_dl.append(float(np.degrees(np.linalg.norm(e[3:]))))
    w_dl.append(float(w))
    qp = q.copy()

export_data["methods"]["dls"] = {
    "q": q_dl,
    "pe": pe_dl,
    "oe": oe_dl,
    "w": w_dl
}

# 3. Newton + NullSpace for multiple alpha values: 0.00, 0.10, 0.30, 0.50
alphas = [0.0, 0.1, 0.3, 0.5]
export_data["alphas"] = {}

for alpha in alphas:
    print(f"Simulating Newton + Nullspace (alpha={alpha:.2f})...")
    q_ns, pe_ns, oe_ns, w_ns, c_ns = [], [], [], [], []
    qp = q_init.copy()
    for i in range(NW):
        q, e, w, c = solve_newton(Td[i], qp, 5, alpha > 0.0, alpha)
        q_ns.append(q.tolist())
        pe_ns.append(float(np.linalg.norm(e[:3]) * 1000))
        oe_ns.append(float(np.degrees(np.linalg.norm(e[3:]))))
        w_ns.append(float(w))
        c_ns.append(float(c))
        qp = q.copy()
    
    # Calculate deviation from midpoint for each joint
    q_ns_arr = np.array(q_ns)
    dev_from_mid = np.abs(q_ns_arr - q_mid)
    mean_dev_deg = np.degrees(dev_from_mid.mean(axis=1)).tolist()
    joint_devs_deg = np.degrees(dev_from_mid).tolist() # list of list of 7 values
    
    export_data["alphas"][f"alpha_{alpha:.2f}"] = {
        "q": q_ns,
        "pe": pe_ns,
        "oe": oe_ns,
        "w": w_ns,
        "cond": c_ns,
        "mean_dev_deg": mean_dev_deg,
        "joint_devs_deg": joint_devs_deg
    }

# Save
output_path = "simulation_results_detailed.json"
with open(output_path, "w") as f:
    json.dump(export_data, f, indent=2)
print(f"Detailed simulation data written to {output_path}")
