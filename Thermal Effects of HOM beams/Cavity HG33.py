# %%
from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3DSteadyState,
)
from ifo_thermal_state.math import composite_newton_cotes_weights

import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
import finesse
from finesse.cymath.homs import HGModes
from finesse.knm import Map
import pickle
from finesse.cymath.zernike import Znm_eval

import matplotlib.animation as animation
from IPython.display import Video
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib.colors import LogNorm

finesse.configure(plotting=True)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


# %%
def update_fem():
    ss_itm.temperature.I_HR.interpolate(I_ITM_HR)
    ss_itm.solve_temperature()
    ss_itm.solve_deformation()

    ss_etm.temperature.I_HR.interpolate(I_ETM_HR)
    ss_etm.solve_temperature()
    ss_etm.solve_deformation()

def get_mask(x, y, ss):
    _, _, mask = ss.evaluate_deformation(x, y, 0.1, meshgrid=True)
    return mask.astype(int)[:, :, 0]


def get_deformation(x, y, ss):
    _, S, mask = ss.evaluate_deformation(x, y, 0.1, meshgrid=True)
    S[~mask] = 0
    return S[:, :, 0, 2]


def get_opd(x, y, ss):
    z = np.linspace(-0.1, 0.1, 5)
    xyz, dT, mask = ss.evaluate_temperature(x, y, z, meshgrid=True)
    dT[~mask] = 0
    # Use better quadrature rule for integration so it requires less z points
    # The depth profile typically looks like a polynomial, so a composite
    # newton cotes here should work well
    weights = composite_newton_cotes_weights(z.size, 5)
    # integrate change in temperature over the optical path and compute how
    # much the optical path depth we accumulate
    dz = z[1] - z[0]
    OPD = (
        model.dndT
        * dz
        * np.sum(
            dT[:, :, :, 0] * weights[None, None, :], axis=2
        )  # weight Z direction and sum
    )
    return OPD


# %% Make the ALIGO test mass
model = make_test_mass_model(
    mesh_function_kwargs={
        "HR_mesh_size": 0.01,
        "AR_mesh_size": 0.01,
        "mesh_algorithm": 6,
    }
)

# %%
ifo = finesse.script.parse(
"""
variable nsilica 1.45
variable Mloss 30u

laser L0 P=2500
s l1 L0.p1 eom.p1

modulator eom 15M 0.1 order=3
s s2 portA=eom.p2 portB=ITMlens.p1

lens ITMlens f=inf
s l2 ITMlens.p2 ITM.p1
m ITM T=0.014 L=Mloss R=1-ITM.L-ITM.T Rc=-2679.93  # -1934
s L ITM.p2 ETM.p1 L=3994
m ETM T=5u L=Mloss R=1-ETM.L-ETM.T Rc=2679.93  # 2245
cav cavARM ITM.p2.o

fd E_itm ITM.p2.i f=0
fd E_etm ETM.p1.i f=0
ad HG33 ETM.p1.i f=0 n=3 m=3
pd Pcirc ITM.p2.o
pd Prefl ITM.p1.o

power_detector_demod_1 pdh ITM.p1.o f=eom.f phase=-0.25014457149285
lock lock_length pdh ETM.phi 0.00041430762864891893 1e-9
"""
)

maxtem = 12
ifo.modes(maxtem=maxtem)

ifo.ITM.phi = 2.45
RcTM = 2651.8
ifo.ITM.Rc = -RcTM
ifo.ETM.Rc = RcTM

ifo.L0.tem(3, 3, 1)
ifo.L0.tem(0, 0, 0)

###### 10% astigmatism ######
# ifo.ITM.phi = -3.15
# deltaS = -10.8e-3
# STM = 1/RcTM
# TM_p = 1/(STM*(1+0.05+deltaS)) 
# TM_m = 1/(STM*(1-0.05+deltaS)) 
# ifo.ITM.Rc = [-TM_p, -TM_m]
# ifo.ETM.Rc = [TM_p, TM_m]
###### 10% astigmatism ######

ifo_0 = ifo.deepcopy()
ifo_0.parse("power_detector_demod_1 pdh_complex ITM.p1.o f=eom.f")
eps = 1e-9 # finite difference step size for gradient calculation
ifo_0.parse(f"xaxis(ITM.phi, lin, -{eps/2}, {eps/2}, 1, relative=True)")
sol = ifo_0.run()
# compute the real and imaginary gradient with respect to the
# cavity length change, then compute the complex argument (angle)
# to get the optimial demodulation phase
opt_demod = np.angle(
    np.gradient(
        sol['pdh_complex'], eps
    ).mean(), # take mean as we need
    deg=True # make sure we compute angle in degrees
)
print(opt_demod)

ifo_0.parse("xaxis(ETM.phi, lin, -5, 5, 400)")
ifo_0.parse(f"pd1 pdh_plot node=ITM.p1.o f=eom.f phase=-0.25014457149285")

out = ifo_0.run()
out.plot(['pdh_plot', 'Pcirc'])

grad = np.gradient(out["pdh_plot"])
idx = np.argmax(np.abs(grad))
# print(out.x1[idx])
grad = (out["pdh_plot"][idx]-out["pdh_plot"][idx-1])/(out.x1[1]-out.x1[0])
print(-1/grad)

# %% Make a simple fabry perot cavity
ifo.beam_trace()  # ABCD trace all the beams through the model
# Set the input laser to be fixed to the currently traced value
# which should be mode matched to the cavity in this state
ifo.L0.p1.o.q = ifo.L0.p1.o.q
out = ifo.run("series(run_locks(display_progress=true), noxaxis())")  # lock the cavity
# out = ifo.run("series(pseudo_lock_cavity(cavARM, mode=[3,3]), noxaxis())")  # lock the cavity
print(ifo.ITM.phi, ifo.ETM.phi)

print("Power circulating", out["noxaxis"]["Pcirc"])
Pout_cold = out["noxaxis"]["Pcirc"]

print(f"Pcirc (cold) with no aperture MASK: {Pout_cold:.2f} W")
# %%
# define the x/y points to evaluate the FEA at generate some distortions to load into finesse
x, y = (
    np.linspace(-0.17, 0.17, 400),
    np.linspace(-0.17, 0.17, 400),
)
# Storage object that the FEA update tools will use to upload
values = SimpleNamespace()
values.out = ifo.run()

def I_ITM_HR(x):
    # interpolates some intensity pattern onto a boundary for the
    # HR surface of the ITM
    HGs = HGModes(ifo.ITM.p2.i.q, ifo.homs)
    # evalute the beam shape over the requested points
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itm"][:, None]
    E = np.sum(a, axis=0)  # Total optical field [sqrt{W}/m^2]
    I = E * E.conj()  # Intensity of optical field [W/m^2]
    return I.real * 0.5e-6


def I_ETM_HR(x):
    # interpolates some intensity pattern onto a boundary for the
    # HR surface of the ETM
    HGs = HGModes(ifo.ETM.p1.i.q, ifo.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etm"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 * 3 / 5

ss_itm = AdvancedLIGOTestMass3DSteadyState(model)
ss_etm = AdvancedLIGOTestMass3DSteadyState(model)

update_fem()

# %%
# ifo_1 = ifo.deepcopy()
# ifo_1.ITM.phi.is_tunable = True
# ifo_1.ETM.phi.is_tunable = True

# ifo_1.ITM.surface_map = Map(x, y, amplitude=get_mask(x, y, ss_itm))
# ifo_1.ETM.surface_map = Map(x, y, amplitude=get_mask(x, y, ss_etm))
# ifo_1.ITMlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, ss_etm))
# ifo_1.L0.P = 2500

# ifo_1.parse("xaxis(ETM.phi, lin, -5, 5, 400)")

# out = ifo_1.run()
# out.plot('pdh');

# grad = np.gradient(out["pdh"])
# idx = np.argmax(np.abs(grad))
# grad = (out["pdh"][idx]-out["pdh"][idx-1])/(out.x1[1]-out.x1[0])
# print(-1/grad)

# %%
ifo.ITM.phi.is_tunable = True
ifo.ETM.phi.is_tunable = True
mask_aper_ITM = get_mask(x, y, ss_itm)
mask_aper_ETM = get_mask(x, y, ss_etm)
ifo.ITM.surface_map = Map(x, y, amplitude=mask_aper_ITM)
ifo.ETM.surface_map = Map(x, y, amplitude=mask_aper_ETM)
ifo.ITMlens.OPD_map = Map(x, y, amplitude=mask_aper_ETM)
ifo.L0.P = 2500

# ifo.run("pseudo_lock_cavity(cavARM, mode=[3,3])")
out = ifo.run("run_locks(display_progress=false)")  # lock the cavity
print(ifo.ITM.phi, ifo.ETM.phi)

values.out = ifo.run("noxaxis()")
Pout_cold_MASK = values.out["Pcirc"]
print(f"Pcirc (cold) with aperture MASK: {Pout_cold_MASK:.2f} W")
print(f"The clipping loss is {1e6*(Pout_cold - Pout_cold_MASK)/Pout_cold:.2f} ppm")

with open('./pkl/HG33_MASKs.pkl', 'wb') as f:
    pickle.dump([mask_aper_ITM, mask_aper_ETM], f)


# %%
def remove_curv(map_data):
    x, y, = (
    np.linspace(-0.17, 0.17, 400),
    np.linspace(-0.17, 0.17, 400),
    )
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    phi = np.arctan2(yy, xx)
    index = r > 0.17
    
    Z20 = Znm_eval(r, phi, 2, 0, 0.17)
    Z20[index] = 0
    
    overlap = np.sum(map_data*Z20)/np.sum(Z20*Z20)
    print(f"The Z20 coefficient is {overlap:.2e} m")

    return map_data - overlap*Z20


# %%
Pcircs = []
P_HG33s = []
ITM_maps = []
ETM_maps = []
ITM_lens_OPDs = []
for i in range(12):
    update_fem()
    ITM_map = -get_deformation(x, y, ss_itm)
    ETM_map = get_deformation(x, y, ss_etm)
    ITM_lens = get_opd(x, y, ss_itm)

    ITM_map = remove_curv(ITM_map)
    ETM_map = remove_curv(ETM_map)
    ITM_lens = remove_curv(ITM_lens)

    ITM_maps.append(ITM_map)
    ETM_maps.append(ETM_map)
    ITM_lens_OPDs.append(ITM_lens)

    ifo.ITM.surface_map = Map(
        x, y, amplitude=get_mask(x, y, ss_itm), opd=ITM_map
    )
    ifo.ETM.surface_map = Map(
        x, y, amplitude=get_mask(x, y, ss_etm), opd=ETM_map
    )
    ifo.ITMlens.OPD_map = Map(
        x, y, amplitude=get_mask(x, y, ss_etm), opd=ITM_lens
    )
    ifo.run("run_locks(display_progress=false)")

    sols = ifo.run("noxaxis()")

    print(values.out["Pcirc"], sols["Pcirc"])
    Pcircs.append(np.sum(abs(values.out["E_itm"]) ** 2))
    P_HG33s.append(abs(values.out["HG33"])**2)
    values.out = sols

Pcircs = np.array(Pcircs)
P_HG33s = np.array(P_HG33s)
ITM_maps = np.array(ITM_maps)
ETM_maps = np.array(ETM_maps)
ITM_lens_OPDs = np.array(ITM_lens_OPDs)

with open('./pkl/cavHG33_compensated.pkl', 'wb') as f:
    pickle.dump([Pcircs, P_HG33s, ITM_maps, ETM_maps, ITM_lens_OPDs], f)

# %%
Pcircs = []
P_HG33s = []
ITM_maps = []
ETM_maps = []
ITM_lens_OPDs = []
# Pcircs = list(Pcircs)
# P_HG33s = list(P_HG33s)
# ITM_maps = list(ITM_maps)
# ETM_maps = list(ETM_maps)
# ITM_lens_OPDs = list(ITM_lens_OPDs)
for i in range(12):
    update_fem()
    ITM_map = -get_deformation(x, y, ss_itm)
    ETM_map = get_deformation(x, y, ss_etm)
    ITM_lens = get_opd(x, y, ss_itm)

    ITM_maps.append(ITM_map)
    ETM_maps.append(ETM_map)
    ITM_lens_OPDs.append(ITM_lens)

    ifo.ITM.surface_map = Map(
        x, y, amplitude=get_mask(x, y, ss_itm), opd=ITM_map
    )
    ifo.ETM.surface_map = Map(
        x, y, amplitude=get_mask(x, y, ss_etm), opd=ETM_map
    )
    ifo.ITMlens.OPD_map = Map(
        x, y, amplitude=get_mask(x, y, ss_etm), opd=ITM_lens
    )
    # sols = ifo.run("series(pseudo_lock_cavity(cavARM, mode=[3,3]), noxaxis())")
    ifo.run("run_locks(display_progress=false)")
    # print(ifo.ITM.phi, ifo.ETM.phi)

    sols = ifo.run("noxaxis()")

    print(values.out["Pcirc"], sols["Pcirc"])
    # Pcircs.append(np.sum(abs(values.out["E_itm"]) ** 2))
    # Pcircs.append(values.out["Pcirc"])
    Pcircs.append(np.sum(abs(values.out["E_itm"]) ** 2))
    P_HG33s.append(abs(values.out["HG33"])**2)
    values.out = sols

Pcircs = np.array(Pcircs)
P_HG33s = np.array(P_HG33s)
ITM_maps = np.array(ITM_maps)
ETM_maps = np.array(ETM_maps)
ITM_lens_OPDs = np.array(ITM_lens_OPDs)

with open('./pkl/cavHG33_astig.pkl', 'wb') as f:
# with open('./pkl/cavHG33.pkl', 'wb') as f:
    pickle.dump([Pcircs, P_HG33s, ITM_maps, ETM_maps, ITM_lens_OPDs], f)

# %%
with open('./pkl/cavHG33.pkl', 'rb') as f:
    Pcircs, P_HG33s, ITM_maps, ETM_maps, ITM_lens_OPDs = pickle.load(f)

with open('./pkl/cavHG33_astig.pkl', 'rb') as f:
    Pcircs_astig, P_HG33s_astig, ITM_maps_astig, ETM_maps_astig, ITM_lens_OPDs_astig = pickle.load(f)

with open('./pkl/cavHG00.pkl', 'rb') as f:
    Pcircs_00, P_HG00s, ITM_maps_00, ETM_maps_00, ITM_lens_OPDs_00 = pickle.load(f)

with open('./pkl/cavHG00_compensated.pkl', 'rb') as f:
    Pcircs_00_comm, P_HG00s_comm, ITM_maps_00_comm, ETM_maps_00_comm, ITM_lens_OPDs_00_comm = pickle.load(f)

with open('./pkl/cavHG33_compensated.pkl', 'rb') as f:
    Pcircs_33_comm, P_HG33s_comm, ITM_maps_33_comm, ETM_maps_33_comm, ITM_lens_OPDs_33_comm = pickle.load(f)

# %%
lw = 2.6
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(P_HG00s/2500, lw=lw, color=CB_color_cycle[8], label=r"$\mathrm{HG}_{0,0}$, Mode")
ax.plot(P_HG00s_comm/2500, lw=lw, color=CB_color_cycle[2], label=r"$\mathrm{HG}_{0,0}$ TCS, Mode")
ax.plot(P_HG33s/2500, lw=lw, color=CB_color_cycle[0], label=r"$\mathrm{HG}_{3,3}$, Mode")
# ax.plot(P_HG33s_comm/2500, lw=lw, color=CB_color_cycle[0], label=r"$\mathrm{HG}_{3,3}$ TCS, Mode")
ax.plot(P_HG33s_astig/2500, lw=lw, color=CB_color_cycle[3], label=r"$\mathrm{HG}_{3,3}$ Astigmatic, Mode")

lw = 3
ax.plot(Pcircs_00/2500, lw=lw, color=CB_color_cycle[5], ls="--", label=r"$\mathrm{HG}_{0,0}$, Total")
ax.plot(Pcircs_00_comm/2500, lw=lw, color=CB_color_cycle[7], ls="--", label=r"$\mathrm{HG}_{0,0}$ TCS, Total")
ax.plot(Pcircs/2500, lw=lw, color=CB_color_cycle[1], ls="--", label=r"$\mathrm{HG}_{3,3}$, Total")
# ax.plot(Pcircs_33_comm/2500, lw=lw, color=CB_color_cycle[1], ls="--", label=r"$\mathrm{HG}_{3,3}$ TCS, Total")
ax.plot(Pcircs_astig/2500, lw=lw, color=CB_color_cycle[4], ls="--", label=r"$\mathrm{HG}_{3,3}$ Astigmatic, Total")

ax.set_xlabel("Iterative Step", fontsize=14, labelpad=5)
ax.set_ylabel(r"ARM Cavity Power Gain", fontsize=15, labelpad=5);

legend = plt.legend(fontsize=12, ncols=2);

# %%

# %%

# %%
lw = 2.6
markersize = 10
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(np.abs(1-P_HG00s/Pcircs_00), lw=lw, color=CB_color_cycle[2], label=r"$\mathrm{HG}_{0,0}$",
            marker="*", markersize=markersize)
ax.semilogy(np.abs(1-P_HG00s_comm/Pcircs_00_comm), lw=lw, color=CB_color_cycle[7], label=r"$\mathrm{HG}_{0,0}$ TCS",
            marker="*", markersize=markersize)
ax.semilogy(np.abs(1-P_HG33s/Pcircs), lw=lw, color=CB_color_cycle[1], label=r"$\mathrm{HG}_{3,3}$",
            marker="*", markersize=markersize)
ax.semilogy(np.abs(1-P_HG33s_astig/Pcircs_astig), lw=lw, color=CB_color_cycle[0], label=r"$\mathrm{HG}_{3,3}$ Astigmatic",
            marker="*", markersize=markersize)


ax.set_xlabel("Iterative Step", fontsize=14, labelpad=5)
ax.set_ylabel(r"ARM Cavity Power Impurity", fontsize=15, labelpad=5);

legend = plt.legend(fontsize=14, ncols=2);


# %%

# %%
def ani_plot(data_ani, save_fn="ITM_lens_OPDs_HG33.mp4"):
    nSeconds = 4
    fps = 3
    
    fig, ax = plt.subplots(figsize=(7.5, 6))
    a = data_ani[0]
    im = ax.imshow(a, interpolation='none', aspect='equal', vmin=a.min(), vmax=a.max(), cmap="jet")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, )#norm=matplotlib.colors.Normalize(vmin=a.min(), vmax=a.max()))
    cbar.set_label(label=r"ITM Lens OPD [m]", fontsize=18, rotation=90, labelpad=1)
    cbar.ax.minorticks_on()
    cbar.ax.tick_params(labelsize=14, rotation=0, pad=1)
    ax.grid(False)
    # ax.set_title("ITM Thermal Lens OPD [m]")
    
    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
    
        im.set_array(data_ani[i])
    
        return [im]
    
    anim = animation.FuncAnimation(
                                   fig, 
                                   animate_func, 
                                   frames = nSeconds * fps,
                                   interval = 1000 / fps, # in ms
                                   )
    
    anim.save(save_fn, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()


# %%
ani_plot(data_ani=ITM_lens_OPDs, save_fn="ITM_lens_OPDs_HG33.mp4")
width = 640
height = 480
Video("ITM_lens_OPDs_HG33.mp4", embed=True, width=width, height=height,)

# %%
ani_plot(data_ani=ITM_lens_OPDs_00, save_fn="ITM_lens_OPDs_HG00.mp4")
width = 640
height = 480
Video("ITM_lens_OPDs_HG00.mp4", embed=True, width=width, height=height,)

# %%
ani_plot(data_ani=ITM_lens_OPDs_00_comm, save_fn="ITM_lens_OPDs_HG00_comm.mp4")
width = 640
height = 480
Video("ITM_lens_OPDs_HG00_comm.mp4", embed=True, width=width, height=height,)

# %%

# %%
with open('./pkl/HG33_MASKs.pkl', 'rb') as f:
    mask_aper_ITM, mask_aper_ETM = pickle.load(f)

with open('./pkl/cavHG33.pkl', 'rb') as f:
    Pcircs, P_HG33s, ITM_maps, ETM_maps, ITM_lens_OPDs = pickle.load(f)

ITM_map_f = ITM_maps[-1]
ETM_map_f = ETM_maps[-1]
ITM_lens_f = ITM_lens_OPDs[-1]
    
ifo_HG33 = ifo.deepcopy()
ifo_HG33.modes(maxtem=20)
for n in range(21):
    for m in range(21-n):
        if not (n==3 and m==3):
            ifo_HG33.parse(f"ad HG{n}_{m} ETM.p1.i f=0 n={n} m={m}")
        
ifo_HG33.ITM.surface_map = Map(
    x, y, amplitude=mask_aper_ITM, opd=ITM_map_f
)
ifo_HG33.ETM.surface_map = Map(
    x, y, amplitude=mask_aper_ETM, opd=ETM_map_f
)
ifo_HG33.ITMlens.OPD_map = Map(
    x, y, amplitude=mask_aper_ETM, opd=ITM_lens_f
)
ifo_HG33.run("run_locks(display_progress=false)")
print(ifo_HG33.ITM.phi, ifo_HG33.ETM.phi)

sols_f = ifo_HG33.run("noxaxis()")

# %%
sols_f_ads = {}
for n in range(21):
    for m in range(21-n):
        if not (n==3 and m==3):
            sols_f_ads[f"HG{n}_{m}"] = abs(sols_f[f"HG{n}_{m}"])
        else:
            sols_f_ads[f"HG{n}_{m}"] = abs(sols_f[f"HG{n}{m}"])

sol_np_arr = np.array(list(sols_f_ads.values()))
Pcirc_f = np.sum(np.abs(sol_np_arr)**2)
print(f"The total final circulating power is {Pcirc_f:.2f} W")
(Pcirc_f-sols_f["Pcirc"])/sols_f["Pcirc"]

# %%
sols_f_2Darr = []
for n in range(11):
    temp = []
    for m in range(11):
        temp.append(np.abs(sols_f_ads[f"HG{n}_{m}"])**2)
    sols_f_2Darr.append(temp)

sols_f_2Darr = np.array(sols_f_2Darr)/Pcirc_f

# %%
minn = 1e-2
maxx = 100 #sols_f_2Darr.max()*100
top_indexes = ["3_3", "1_5", "5_1", "3_9", "5_7", "7_5", "9_3"]
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(sols_f_2Darr*100, cmap="jet", norm=LogNorm(vmin=minn, vmax=maxx),)

ax.set_xticks(np.arange(11))
ax.set_yticks(np.arange(11))

top_values = []
for index, top in enumerate(top_indexes):
    n, m = top.split("_")
    n = int(n)
    m = int(m)
    color = CB_color_cycle[2] if n == 3 and m == 3 else CB_color_cycle[7] 
    top_val = np.abs(sols_f_ads[f"HG{n}_{m}"])**2/Pcirc_f*100
    top_values.append(top_val)
    ax.text(n, m, f"{top_val:.1f}", fontsize=11, verticalalignment='center',
            horizontalalignment='center', color=color, fontweight="bold")

ax.set_xlabel("Mode Index n", fontsize=14)
ax.set_ylabel("Mode Index m", fontsize=16);

plt.grid(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = fig.colorbar(im, cax=cax, norm=matplotlib.colors.LogNorm(vmin=0, vmax=1))
cbar.set_label(label=r"Power (%)", fontsize=18, rotation=90, labelpad=1)
cbar.ax.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())

# cbar.ax.minorticks_on()
cbar.ax.tick_params(labelsize=14, rotation=0, pad=1);
print(f"They summed to {np.sum(top_values):.2f} %")

# %%

# %%
ifo_HG33 = ifo.deepcopy()
ifo_HG33.modes(maxtem=20)
for n in range(21):
    for m in range(21-n):
        if (n+m==6 or n+m==12):
            ifo_HG33.parse(f"ad HG{n}_{m} ETM.p1.i f=0 n={n} m={m}")
        
ifo_HG33.ITM.surface_map = Map(
    x, y, amplitude=mask_aper_ITM, opd=ITM_map_f
)
ifo_HG33.ETM.surface_map = Map(
    x, y, amplitude=mask_aper_ETM, opd=ETM_map_f
)
ifo_HG33.ITMlens.OPD_map = Map(
    x, y, amplitude=mask_aper_ETM, opd=ITM_lens_f
)

out_f_scan = ifo_HG33.run("xaxis(ETM.phi, lin, -5, 5, 400)")

# %%
lw = 3
top_indexes = ["3_3", "1_5", "5_1", "3_9", "5_7", "7_5", "9_3"]
colors = [4, 0, 1, 2, 3, 5, 7,]
lss = ["-", "-", "--", "-", "-", "--", "--"]
fig, ax = plt.subplots(figsize=(8, 6))

for index, top in enumerate(top_indexes):
    n, m = top.split("_")
    n = int(n)
    m = int(m)
    ax.semilogy(out_f_scan.x1, np.abs(out_f_scan[f"HG{n}_{m}"])**2, lw=lw, color=CB_color_cycle[colors[index]], 
                ls=lss[index], label=rf"$\mathrm{{HG}}_{{{n},{m}}}$ Mode")

ax.set_xlabel("ETM Phi [deg]", fontsize=14, labelpad=5)
ax.set_ylabel(r"Circulating Power [W]", fontsize=15, labelpad=5);
ax.yaxis.set_minor_locator(plt.NullLocator())

legend = plt.legend(fontsize=12, ncols=2);

# %%

# %%
