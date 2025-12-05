import numpy as np
import plotly.graph_objects as go
from scipy.special import sph_harm, genlaguerre, factorial


#------------------------------------
# Bohr radius
#------------------------------------
a0 = 1.0


#------------------------------------
# General radial function
#------------------------------------
def R_nl(n, l, r):
    rho = 2*r/(n*a0)
    norm = np.sqrt((2/(n*a0))**3 * factorial(n-l-1)/(2*n*factorial(n+l)))
    laguerre = genlaguerre(n-l-1, 2*l+1)(rho)
    return norm * np.exp(-rho/2) * rho**l * laguerre


#------------------------------------
# General orbital
#------------------------------------
def psi_nlm(n, l, m, R, theta, phi):
    radial = R_nl(n, l, R)
    angular = sph_harm(m, l, phi, theta)
    return radial * angular


#------------------------------------
# Grid Design
#------------------------------------
grid_size = 100
extent = 15.0
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
z = np.linspace(-extent, extent, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
R = np.sqrt(X**2 + Y**2 + Z**2)
theta = np.arccos(np.divide(Z, R, out=np.zeros_like(R), where=R!=0))
phi = np.arctan2(Y, X)


#------------------------------------
# Orbitals to include
#------------------------------------
orbitals = [
    {"label": "1s (n=1,l=0,m=0)", "n":1, "l":0, "m":0},
    {"label": "2p (n=2,l=1,m=0)", "n":2, "l":1, "m":0},
    {"label": "2p (n=2,l=1,m=1)", "n":2, "l":1, "m":1},
    {"label": "3d (n=3,l=2,m=0)", "n":3, "l":2, "m":0},
    {"label": "3d (n=3,l=2,m=2)", "n":3, "l":2, "m":2},
]



#------------------------------------
# Default orbital
#------------------------------------
n, l, m = 1, 0, 0
psi = psi_nlm(n, l, m, R, theta, phi)
rho_density = np.abs(psi)**2
rho_density /= rho_density.max()
rho_phase = psi.real
rho_phase /= np.max(np.abs(rho_phase))


#------------------------------------
# Initial figure (density view)
#------------------------------------
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=rho_density.flatten(),
    isomin=0.1,
    isomax=1.0,
    opacity=0.1,
    surface_count=25,
    colorscale='RdBu',
    caps=dict(x_show=False, y_show=False, z_show=False)
))


#------------------------------------
# Dropdown for orbitals
#------------------------------------
buttons_orb = []
for orb in orbitals:
    psi = psi_nlm(orb["n"], orb["l"], orb["m"], R, theta, phi)
    rho_density = np.abs(psi)**2
    rho_density /= rho_density.max()
    rho_phase = psi.real
    rho_phase /= np.max(np.abs(rho_phase))
    buttons_orb.append(dict(
        label=orb["label"],
        method="update",
        args=[{"value": [rho_density.flatten()]}]  # default density view
    ))



#------------------------------------
# Toggle between density and phase
#------------------------------------
buttons_view = [
    dict(label="Density (|ψ|²)", method="update",
         args=[{"value": [rho_density.flatten()],
                "colorscale": ['Viridis']}]),
    dict(label="Phase (Re[ψ])", method="update",
         args=[{"value": [rho_phase.flatten()],
                "colorscale": ['RdBu']}])
]


#------------------------------------
# Layout
#------------------------------------
fig.update_layout(
    title="Hydrogen Orbitals Visualization",
    scene=dict(aspectmode="cube"),
    updatemenus=[
        dict(buttons=buttons_orb, direction="down", showactive=True, x=0.1, y=1.1),
        dict(buttons=buttons_view, direction="down", showactive=True, x=0.35, y=1.1)
    ]
)

fig.show()


#------------------------------------
# Note: Blue indicates High Density and Red indicates Low Density
#------------------------------------