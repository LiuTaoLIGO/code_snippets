# %%
from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3DSteadyState,
)
from ifo_thermal_state.math import composite_newton_cotes_weights
from ifo_thermal_state.plotting import plot_deformation, plot_mesh, plot_temperature

import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
import finesse
from finesse.cymath.homs import HGModes
from finesse.cymath.zernike import Znm_eval
from IPython.display import IFrame
import pickle

finesse.configure(plotting=True)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# %%
model = make_test_mass_model(
    mesh_function_kwargs={
        "HR_mesh_size": 0.01,
        "AR_mesh_size": 0.01,
        "mesh_algorithm": 0,
    }
)
# plot_mesh(model.msh, model.cell_markers)
model.dndT

# %%
ifo = finesse.script.parse(
f"""
l l1 P=1
m m1
link(l1, m1)
modes([[0,0],[3,3]])
fd E m1.p1.i f=0
gauss g1 l1.p1.o w0={0.17*.263} z=0
"""
)
ifo.l1.tem(3, 3, 1)
ifo.l1.tem(0, 0, 0)

out = ifo.run()
ifo.homs
# %%
qx, qy = ifo.m1.p1.i.q
amplitudes = out["E"]
print(amplitudes[:, None])
print(ifo.homs)
x = y = np.linspace(0, 0.1)
HGs = HGModes((qx, qy), ifo.homs)

# %%
values = SimpleNamespace()
values.P_coat = 1

def I_HR(x):
    a = HGs.compute_points(x[0], x[1]) * amplitudes[:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * values.P_coat

ss = AdvancedLIGOTestMass3DSteadyState(model)

# %%
ss.temperature.I_HR.interpolate(I_HR)
ss.solve_temperature()
ss.solve_deformation()

# plot_deformation(ss.deformation.V, ss.deformation.solution)

# %%
# Plotting temperature field
# plot_temperature(ss.temperature.V, ss.temperature.solution)

# %%
x, y, z = (
    np.linspace(-0.17, 0.17, 200),
    np.linspace(-0.17, 0.17, 200),
    [-0.1, 0.1],
)

# %%
xyz, dT, mask = ss.evaluate_temperature(x, y, z, meshgrid=True)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, title in zip(range(2), ["AR", "HR"]):
    _mask = mask[:, :, i]
    plt.sca(axs[i])
    plt.contourf(
        x,
        y,
        dT[:, :, i, 0],
        20,
    )
    plt.gca().set_aspect("equal")
    cb0 = plt.colorbar()
    cb0.ax.set_ylabel("Temperature [K]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)

plt.tight_layout()

# %%
xyz, S, mask = ss.evaluate_deformation(x, y, z, meshgrid=True)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, title in zip(range(2), ["AR", "HR"]):
    _mask = mask[:, :, i]
    plt.sca(axs[i])
    plt.contourf(
        x,
        y,
        1e9 * (S[:, :, i, 2] - S[_mask, i, 2].min()),
        20,
    )
    plt.gca().set_aspect("equal")
    cb0 = plt.colorbar()
    cb0.ax.set_ylabel("Surface deformation [nm]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)

plt.tight_layout()

# %% 3D integration
x, y, z = (
    np.linspace(-0.17, 0.17, 1000),
    np.linspace(-0.17, 0.17, 1000),
    np.linspace(-0.1, 0.1, 7),
)
# %%
xyz, dT, mask = ss.evaluate_temperature(x, y, z, meshgrid=True)
dT = np.nan_to_num(dT) # convert all nans to zero
# %%
# Use better quadrature rule for integratiopn
weights = composite_newton_cotes_weights(z.size, 7)
dz = z[1] - z[0]
OPD_HOM = (
    model.dndT
    * dz
    * np.sum(
        dT[:, :, :, 0] * weights[None, None, :], axis=2
    )  # weight Z direction and sum  OPD_HOM
)

plt.contourf(
    x,
    y,
    OPD_HOM / 1e-6,
    20,
)
plt.gca().set_aspect("equal")
cb0 = plt.colorbar()
cb0.ax.set_ylabel("Optical path depth [um]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Substrate optical path depth");
# %%

# %%
def Zernike_decom(OPD_data, max_n=7, a0=0.17):
    x, y, = (
    np.linspace(-0.17, 0.17, 1000),
    np.linspace(-0.17, 0.17, 1000),
    )
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    phi = np.arctan2(yy, xx)
    index = r > 0.17
    texts = []
    overlap_zers = []
    idx_tot = 0
    for n in range(max_n):
        for m in np.arange(-n, n+1, 2):
            idx_tot += 1
            Znm = Znm_eval(r, phi, n, m, a0)
            Znm[index] = 0
            
            overlap = np.sum(OPD_data*Znm)/np.sum(Znm*Znm)
            overlap_zers.append(overlap)
            if m < 0:
                text = rf"$\mathrm{{Z}}^{{-{abs(m)}}}_{{{n}}}$"
            else:
                text = rf"$\mathrm{{Z}}^{{{m}}}_{{{n}}}$"
            texts.append(text)

    return idx_tot, texts, np.array(overlap_zers)

idx_tot, texts, overlap_zers_HOM = Zernike_decom(OPD_HOM)
circle = plt.Circle((500, 500), 500, color='gray', linewidth=3, fill=False);
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2+yy**2)
phi = np.arctan2(yy, xx)
index = r > 0.17
Znm_data = Znm_eval(r, phi, 4, 4, 1)
Znm_data[index] = np.nan
plt.imshow(Znm_data, cmap="jet")
plt.gca().add_patch(circle);

# %%
lw = 2
fig, ax1 = plt.subplots(figsize=(7, 5))

ax1.plot(range(idx_tot), overlap_zers_HOM, lw=lw, color=CB_color_cycle[0], marker="*", markersize=12)

ax1.set_xlabel("Zernike Polynomials", fontsize=14, labelpad=5)
ax1.set_ylabel(r"Zernike Coefficients [m]", fontsize=15, labelpad=5)
ax1.set_xticks(range(idx_tot))
ax1.set_xticklabels(texts, rotation=40);


# %%
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
# fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# # plot the same data on both axes
# ax1.plot(range(idx_tot), overlap_zers_HOM, lw=lw, color=CB_color_cycle[0], marker="*", markersize=12)
# ax2.plot(range(idx_tot), overlap_zers_HOM*1e6, lw=lw, color=CB_color_cycle[0], marker="*", markersize=12)

# # zoom-in / limit the view to different portions of the data
# ax2.set_ylim(-27, -24.8)  # outliers only
# ax1.set_ylim(-1e-5, 1e-5)  # most of the data

# ax2.set_yticks([-27, -26, -25])

# # hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()

# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# ax2.set_xlabel("Zernike Polynomials", fontsize=14, labelpad=5)
# ax1.set_ylabel(r"Zernike Coefficients [m]", fontsize=15, labelpad=12, y=0)
# ax2.set_xticks(range(idx_tot))
# ax2.set_xticklabels(texts, rotation=40);

# %%
def radial_profile(mmap, center):
    freqs = np.fft.fftfreq(mmap.shape[0], d=0.17/1000)
    stepfreq = freqs[1] - freqs[0]
    
    mmap_fourier = np.fft.fftshift(np.fft.fft2(mmap, norm='forward'))
    mmap_fourier = np.abs(mmap_fourier)**2

    y, x = np.indices((mmap_fourier.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), mmap_fourier.ravel())*stepfreq
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return np.unique(r)*stepfreq, radialprofile**0.5


# %%

# %%
def OPD_HGModes(n, m):
    model = make_test_mass_model(
    mesh_function_kwargs={
        "HR_mesh_size": 0.01,
        "AR_mesh_size": 0.01,
        "mesh_algorithm": 0,
    }
    )

    ifo = finesse.script.parse(
    f"""
    l l1 P=1
    m m1
    link(l1, m1)
    fd E m1.p1.i f=0
    gauss g1 l1.p1.o w0={0.17*.380} z=0
    """
    )

    if not(n==0 and m==0):
        ifo.parse(f"modes([[0,0],[{n},{m}]])")
        ifo.l1.tem(0, 0, 0)
        ifo.l1.tem(n, m, 1)
        if n==3 and m==3:
            ifo.remove(ifo.g1)
            ifo.parse(f"gauss g1 l1.p1.o w0={0.17*.263} z=0")
        
    
    out = ifo.run()
    qx, qy = ifo.m1.p1.i.q
    amplitudes = out["E"]
    x = y = np.linspace(0, 0.1)
    HGs = HGModes((qx, qy), ifo.homs)

    values = SimpleNamespace()
    values.P_coat = 1
    
    def I_HR(x):
        a = HGs.compute_points(x[0], x[1]) * amplitudes[:, None]
        E = np.sum(a, axis=0)
        I = E * E.conj()
        return I.real * values.P_coat
    
    ss = AdvancedLIGOTestMass3DSteadyState(model)

    ss.temperature.I_HR.interpolate(I_HR)
    ss.solve_temperature()
    ss.solve_deformation()

    x, y, z = (
        np.linspace(-0.17, 0.17, 1000),
        np.linspace(-0.17, 0.17, 1000),
        np.linspace(-0.1, 0.1, 7),
    )

    xyz, dT, mask = ss.evaluate_temperature(x, y, z, meshgrid=True)
    dT = np.nan_to_num(dT)
    weights = composite_newton_cotes_weights(z.size, 7)
    dz = z[1] - z[0]
    OPD_data = (
        model.dndT
        * dz
        * np.sum(
            dT[:, :, :, 0] * weights[None, None, :], axis=2
        )  # weight Z direction and sum  OPD_HOM
    )

    idx_tot, texts, overlap_zers_HOM = Zernike_decom(OPD_data, max_n=13)
    freqAxis, radialProfile = radial_profile(OPD_data, (500, 500))
    
    return freqAxis, radialProfile, idx_tot, texts, overlap_zers_HOM

# %%
freqAxis_0, radialProfile_0, idx_tot, texts, overlap_zers_0 = OPD_HGModes(0, 0)
freqAxis_3, radialProfile_3, idx_tot, texts, overlap_zers_3 = OPD_HGModes(3, 3)


with open('./pkl/OPDHG33.pkl', 'wb') as f:
    pickle.dump([freqAxis_3, radialProfile_3, idx_tot, texts, overlap_zers_3], f)

with open('./pkl/OPDHG00.pkl', 'wb') as f:
    pickle.dump([freqAxis_0, radialProfile_0, idx_tot, texts, overlap_zers_0], f)

# %%
with open('./pkl/OPDHG33.pkl', 'rb') as f:
    freqAxis_3, radialProfile_3, idx_tot, texts, overlap_zers_3 = pickle.load(f)

with open('./pkl/OPDHG00.pkl', 'rb') as f:
    freqAxis_0, radialProfile_0, idx_tot, texts, overlap_zers_0 = pickle.load(f)

# %%
fig, ax1 = plt.subplots(figsize=(7, 5))

ax1.loglog(freqAxis_0, radialProfile_0, lw=1.5, color=CB_color_cycle[0], label=r"$\mathrm{HG}_{0,0}$")
ax1.loglog(freqAxis_3, radialProfile_3, lw=1.5, color=CB_color_cycle[1], label=r"$\mathrm{HG}_{3,3}$")

axins = ax1.inset_axes([0.2, 0.08, 0.4, 0.3],)
axins.loglog(freqAxis_0, radialProfile_0, lw=1.5, color=CB_color_cycle[0], )
axins.loglog(freqAxis_3, radialProfile_3, lw=1.5, color=CB_color_cycle[1], )
axins.set_xlim([5, 50])
axins.set_ylim([4e-9, 1e-6])

axins.set_xticks([10, 20, 30, 40, 50])
axins.set_xticklabels([10, 20, 30, 40, 50])
axins.minorticks_off()

ax1.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6)

ax1.set_xlabel("Spatial Frequency [1/m]", fontsize=14, labelpad=5)
ax1.set_ylabel(r"Height [m/sqrt(m)]", fontsize=15, labelpad=5);

legend = plt.legend(fontsize=16, ncols=1);

# %%
lw = 2
indexx = 78
xticks = np.arange(idx_tot)

fig, ax1 = plt.subplots(figsize=(16, 5))

ax1.plot(xticks[:indexx], overlap_zers_0[:indexx], lw=lw, color=CB_color_cycle[0], marker="*", markersize=12, label=r"$\mathrm{HG}_{0,0}$")
ax1.plot(xticks[:indexx], overlap_zers_3[:indexx], lw=lw, color=CB_color_cycle[1], marker="*", markersize=12, label=r"$\mathrm{HG}_{3,3}$")

ax1.set_xlabel("Zernike Polynomials", fontsize=14, labelpad=5)
ax1.set_ylabel(r"Zernike Coefficients [m]", fontsize=15, labelpad=5)
ax1.set_xticks(xticks[:indexx])
ax1.set_xticklabels(texts[:indexx], rotation=40)

legend = plt.legend(fontsize=16, ncols=1, loc="upper left");

# %%
idx_4th = 0
idx_4ths = []
for n in range(13):
    for m in np.arange(-n, n+1, 2):
        if m == 4:
            idx_4ths.append(idx_4th)
        idx_4th += 1
idx_4ths

# %%
idx_8th = 0
idx_8ths = []
for n in range(13):
    for m in np.arange(-n, n+1, 2):
        if m == 8:
            idx_8ths.append(idx_8th)
        idx_8th += 1
idx_8ths

# %%
indexx = 91 #78
lw = 1.8
relative_3_0 = (overlap_zers_3-overlap_zers_0)/overlap_zers_0
# relative_3_0 = np.where(np.abs(relative_3_0)>10, 0, relative_3_0)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 5))

# Plot data on each subplot
ax1.plot(xticks[:indexx], relative_3_0[:indexx], lw=lw, color=CB_color_cycle[0], marker="*", markersize=12)
ax2.plot(xticks[:indexx], relative_3_0[:indexx], lw=lw, color=CB_color_cycle[0], marker="*", markersize=12)
ax3.plot(xticks[:indexx], relative_3_0[:indexx], lw=lw, color=CB_color_cycle[0], marker="*", markersize=12)

for ax in [ax1, ax2, ax3]:
    ax.plot(xticks[idx_4ths], relative_3_0[idx_4ths], lw=lw, color=CB_color_cycle[2], marker="X", markersize=15, linestyle='None',)
    ax.plot(xticks[idx_8ths], relative_3_0[idx_8ths], lw=lw, color=CB_color_cycle[7], marker="X", markersize=15, linestyle='None',)

lw=3
ax.plot([], [], lw=lw, color=CB_color_cycle[2], marker="X", markersize=12, label=r"$\mathrm{{Z}}^{4}_{n}$")
ax.plot([], [], lw=lw, color=CB_color_cycle[7], marker="X", markersize=12, label=r"$\mathrm{{Z}}^{8}_{n}$")

# Set limits and ticks for each subplot
ax1.set_ylim(55, 110)  # outliers only
ax2.set_ylim(-25, 21)  # most of the data
# ax2.set_ylim(-5, 5)  # most of the data
ax3.set_ylim(-100, -90)  # outliers only

# Hide the spines between ax1 and ax2, ax2 and ax3
ax1.spines.bottom.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)

ax2.tick_params(axis='x', which='both', bottom=False) 
ax2.xaxis.grid(True)
ax3.spines.top.set_visible(False)

# Add lines to connect the broken axes
d = 0.008  # Size of the diagonal line
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, )
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax3.transAxes)
ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
ax3.plot((-d, +d), (1-d, 1+d), **kwargs)  

ax3.set_xticks(xticks[:indexx])
ax3.set_xticklabels(texts[:indexx], rotation=40);
ax3.set_yticks([-100, -95, -90])

ax3.set_xlabel("Zernike Polynomials", fontsize=14, labelpad=5)
ax2.set_ylabel(r"Relative Zernike Coefficients", fontsize=15, labelpad=8, y=0.4)
legend = plt.legend(fontsize=16, ncols=1, loc="upper left");

# %%
x, y, z = (
    np.linspace(-0.17, 0.17, 1000),
    np.linspace(-0.17, 0.17, 1000),
    np.linspace(-0.1, 0.1, 7),
)
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2+yy**2)
phi = np.arctan2(yy, xx)
index = r > 0.17
zernikes = [(4, 4), (6, 4), (8, 4), (8, 8), (10, 8), (12, 8),]
Znm_data_plot = []
texts_plot = []
for n, m in zernikes:  
    Znm_data = Znm_eval(r, phi, n, m, 1)
    Znm_data[index] = np.nan
    Znm_data_plot.append(Znm_data)
    if m < 0:
        text = rf"$\mathrm{{Z}}^{{-{abs(m)}}}_{{{n}}}$"
    else:
        text = rf"$\mathrm{{Z}}^{{{m}}}_{{{n}}}$"
    texts_plot.append(text)

# %%
fig, ((ax0, ax1, ax2,), (ax3, ax4, ax5)) = plt.subplots(figsize=(11, 5), nrows=2, ncols=3)

for idx, data in enumerate(Znm_data_plot):
    circle = plt.Circle((500, 500), 500, color='white', linewidth=3, fill=False)
    eval(f"ax{idx}").imshow(data, cmap="jet")
    eval(f"ax{idx}").add_patch(circle)
    
    eval(f"ax{idx}").axis("off")
    color = "black"
    eval(f"ax{idx}").text(0.95, 0., texts_plot[idx], transform=eval(f"ax{idx}").transAxes, fontsize=13,
                verticalalignment='bottom', horizontalalignment='right',
                color=color, fontname="Courier")

# %%

# %%
IFrame(src='./figures/HG33LG22_lossperzernike.pdf', width=880, height=550)

# %%

# %%
